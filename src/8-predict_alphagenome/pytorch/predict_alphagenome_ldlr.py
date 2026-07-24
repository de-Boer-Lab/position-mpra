"""
AlphaGenome sliding-window (random-insertion) SNV predictions for LDLR promoter
variants, 1 bp-resolution HepG2 RNA-seq, using the PyTorch model.

Model input / output
--------------------
  Input  : 131,072 bp  (2^17) -- fixed by the model architecture, NOT 1 Mb.
  Output : rna_seq at 1 bp resolution (resolutions=(1,), preds.rna_seq[1]) --
           131,072 values per track, HepG2 track only.

Variants + window come from the TSV built by create_variant_data_LDLR.ipynb:
  /scratch/st-cdeboer-1/sambina/position_mpra/outputs/8-aphagenome/LDLR/
    LDLR_c-121T_C_c-120C_T_window.tsv
One shared 131,072 bp window per variant (52,429 bp upstream / 78,643 bp downstream
of the c.-118 anchor); columns: variant, chrom, window_start, window_end, var_pos,
var_pos_in_window, anchor_pos_in_window, atg_pos_in_window, ref, alt.

Sliding-window (random-insertion) design
-----------------------------------------
For each variant, walk K = 0, 32, 64, ... up to the largest multiple of 32 that keeps
the variant inside the window (K <= var_pos_in_window -- going further would delete
the variant from the window, which is why this can't reach anywhere near 100 kb; the
window's upstream flank is only 52,429 bp to begin with):

  [ real genomic, (window_start + K) .. anchor   -- shrinks as K grows            ]
  [ K random bp inserted immediately after the anchor                            ]
  [ real genomic, anchor+1 .. window_end          -- ALWAYS the same 78,642 bp,
                                                      never truncated             ]

Removing K bp from the far upstream edge and inserting K random bp right after the
anchor keeps the total length fixed at 131,072 bp automatically (no separate
truncation needed), and leaves the downstream side -- including the ATG and the full
LDLR gene body -- byte-for-byte identical in every window at a FIXED offset. Only the
upstream/promoter side erodes and gets diluted with random sequence as K grows. The
variant's offset within the window shrinks by K each step.

Saved per variant -- {out_dir}/{variant}/:
  ref.npy, alt.npy
      each a memory-mapped (n_windows, 131_072) float16 array (HepG2 track, forward
      strand only). ve = ref - alt is cheap to recompute later, so it isn't stored.
      The reverse-complement forward pass is skipped entirely (not just unsaved) --
      there's no ref_rc/alt_rc/ve_rc here, and re-adding them later means rerunning
      the model, since discarded-not-saved output can't be recovered otherwise.
  done_mask.npy   (n_windows,) bool memmap -- lets a re-run resume mid-sweep
  meta.tsv        one row per window: window_idx, K, var_pos_in_window (post-shift),
                  plus the constant window/variant info from the input TSV

Why memmapped .npy instead of the old per-window JSON scheme: JSON serialization
(a) upcasts float32 -> Python float64 before writing and (b) encodes floats as ASCII
text, so it is both less precise and dramatically larger/slower than binary float32.
A single preallocated memmap per array avoids ever holding the full prediction matrix
in RAM, avoids the per-window file-open overhead of thousands of small JSON files, and
is directly usable later via `np.load(path, mmap_mode='r')` without loading it all in.
Storage: ~1,639 windows x 2 arrays x 131,072 x 2 bytes (float16) ≈ 0.8 GB per variant.
Compute: dropping the revcomp forward pass halves the batch size (2 sequences instead
of 4 per window), roughly halving runtime vs. computing-and-discarding ref_rc/alt_rc.
float16 has ~3 decimal digits of precision -- fine for plotting, but re-check this if
these values ever feed into something needing finer numerical precision.

Parallelizing across a SLURM array
-----------------------------------
Two-phase, to avoid a race: if the array's tasks all start around the same time and
ref.npy/alt.npy/done_mask.npy don't exist yet, they'd race to create them with
mode="w+" and clobber each other. So:

  1. Setup (once, no GPU needed): creates the files and writes meta.tsv.
       python predict_alphagenome_ldlr.py --setup_only

  2. Compute (array job, GPU): each task only opens files in mode="r+" (never
     creates them) and writes to its own disjoint slice of window indices --
     safe for concurrent tasks since the byte ranges never overlap.
       python predict_alphagenome_ldlr.py --task_id $SLURM_ARRAY_TASK_ID --num_tasks 20
     --task_id/--num_tasks default to the SLURM_ARRAY_TASK_ID/SLURM_ARRAY_TASK_COUNT
     env vars when omitted, so the array script doesn't need to pass them explicitly.

See run_predict_alphagenome_ldlr_setup.sh and run_predict_alphagenome_ldlr_array.sh.

Usage
-----
  python predict_alphagenome_ldlr.py --setup_only
  python predict_alphagenome_ldlr.py --task_id 0 --num_tasks 20
  python predict_alphagenome_ldlr.py --variant "c.-121T>C" --max_windows 5 --device cuda
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.named_outputs import TrackMetadataCatalog
from pyfaidx import Fasta
from tqdm import tqdm

# ── Fixed paths / constants ────────────────────────────────────────────────
DATA_DIR = "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/8-aphagenome"
REF_FASTA = "/scratch/st-cdeboer-1/sambina/reference_genome/hg38.fa"
WEIGHTS = f"{DATA_DIR}/pytorch/model_all_folds.safetensors"
OUTPUT_DIR = f"{DATA_DIR}/pytorch/predictions_ldlr_1bp"
VARIANTS_TSV = f"{DATA_DIR}/LDLR/LDLR_c-121T_C_c-120C_T_window.tsv"

MODEL_LEN = 131_072  # model input length (bp), fixed by the model architecture
STEP = 32  # bp increment for the K sweep
DTYPE = np.float16  # only ref/alt are saved, so precision loss here just affects plots

# One-hot encoding lookup table
_LUT = np.zeros(256, dtype=np.int8)
for _i, _b in enumerate("ACGT"):
    _LUT[ord(_b)] = _i

# Deterministic random pool for the inserted bp (sliced to whatever K is needed).
_BASES = np.array(list("ACGT"))
_MAX_K = 60_000  # comfortably above the largest K any LDLR variant here will need
RAND_POOL = "".join(_BASES[np.random.default_rng(seed=42).integers(0, 4, size=_MAX_K)])


# ── Sequence utilities ─────────────────────────────────────────────────────


def onehot(seq: str) -> torch.Tensor:
    """DNA string → (1, L, 4) float32 NLC tensor  (A=0 C=1 G=2 T=3)."""
    arr = _LUT[np.frombuffer(seq.encode(), dtype=np.uint8)]
    return torch.from_numpy(np.eye(4, dtype=np.float32)[arr]).unsqueeze(0)


def build_window(genomic: str, anchor_offset: int, k: int) -> str:
    """Remove the first k bp, insert k random bp right after the anchor.

    Total length is unchanged: (anchor_offset + 1 - k) real bp, then k random bp,
    then the untouched (len(genomic) - anchor_offset - 1) bp downstream of the anchor.
    """
    left_kept = genomic[k : anchor_offset + 1]
    right_unchanged = genomic[anchor_offset + 1 :]
    seq = left_kept + RAND_POOL[:k] + right_unchanged
    assert len(seq) == MODEL_LEN, f"sequence length {len(seq)} != {MODEL_LEN}"
    return seq


# ── Model loading ──────────────────────────────────────────────────────────


def load_model(device: str) -> AlphaGenome:
    model = AlphaGenome.from_pretrained(WEIGHTS, device=device)
    catalog = TrackMetadataCatalog.load_builtin(organism=0)
    model.set_track_metadata_catalog(catalog)
    model.eval()
    return model


# ── Partitioning across a SLURM array ───────────────────────────────────────


def partition_indices(n_windows: int, task_id: int, num_tasks: int) -> list:
    """Split range(n_windows) into num_tasks contiguous, near-equal chunks; return task_id's."""
    assert 0 <= task_id < num_tasks, f"task_id {task_id} out of range [0, {num_tasks})"
    chunk = -(-n_windows // num_tasks)  # ceil division
    start = min(task_id * chunk, n_windows)
    end = min(start + chunk, n_windows)
    return list(range(start, end))


def k_values_for(row: pd.Series, max_windows: int = None) -> list:
    var_offset = int(row["var_pos_in_window"])
    k_values = list(range(0, (var_offset // STEP) * STEP + 1, STEP))
    if max_windows is not None:
        k_values = k_values[:max_windows]
    return k_values


# ── Setup: create output files + meta.tsv (run once, no GPU, no race) ──────


def setup_variant(row: pd.Series, out_dir: str, max_windows: int = None) -> int:
    """Preallocate ref.npy/alt.npy/done_mask.npy and write meta.tsv. Idempotent.

    Must complete before any compute task (run_variant_chunk) starts -- compute
    tasks only ever open these files in mode="r+", never create them, specifically
    so concurrent SLURM array tasks can't race to create/truncate the same file.
    """
    os.makedirs(out_dir, exist_ok=True)

    chrom = row["chrom"]
    window_start = int(row["window_start"])
    window_end = int(row["window_end"])
    var_pos = int(row["var_pos"])
    var_offset = int(row["var_pos_in_window"])
    anchor_offset = int(row["anchor_pos_in_window"])
    atg_offset = int(row["atg_pos_in_window"])
    ref_allele = row["ref"].upper()
    alt_allele = row["alt"].upper()

    k_values = k_values_for(row, max_windows)
    n_windows = len(k_values)
    print(
        f"  [setup] window={chrom}:{window_start}-{window_end}  var_offset={var_offset}  "
        f"anchor_offset={anchor_offset}  atg_offset={atg_offset} (fixed across the sweep)  "
        f"→ {n_windows} windows (K=0..{k_values[-1]}, step={STEP})"
    )

    for name in ("ref", "alt"):
        path = os.path.join(out_dir, f"{name}.npy")
        if os.path.exists(path):
            arr = np.lib.format.open_memmap(path, mode="r+")
            assert arr.shape == (n_windows, MODEL_LEN), (
                f"{name}.npy shape mismatch, delete {out_dir} to restart"
            )
        else:
            np.lib.format.open_memmap(path, mode="w+", dtype=DTYPE, shape=(n_windows, MODEL_LEN))

    done_mask_path = os.path.join(out_dir, "done_mask.npy")
    if os.path.exists(done_mask_path):
        done_mask = np.lib.format.open_memmap(done_mask_path, mode="r+")
        assert done_mask.shape == (n_windows,)
    else:
        done_mask = np.lib.format.open_memmap(
            done_mask_path, mode="w+", dtype=bool, shape=(n_windows,)
        )
        done_mask[:] = False

    meta_rows = [
        {
            "window_idx": i,
            "K": k,
            "var_pos_in_window": var_offset - k,
            "var_pos_hg38": var_pos,
            "anchor_pos_in_window": anchor_offset - k,
            "atg_pos_in_window": atg_offset,  # fixed -- ATG sits downstream of the anchor,
            # untouched by the K-sweep, so its offset never shifts (unlike var/anchor)
            "window_start": window_start,
            "window_end": window_end,
            "chrom": chrom,
            "ref": ref_allele,
            "alt": alt_allele,
        }
        for i, k in enumerate(k_values)
    ]
    pd.DataFrame(meta_rows).to_csv(os.path.join(out_dir, "meta.tsv"), sep="\t", index=False)
    print(f"  [setup] {n_windows} windows, files ready → {out_dir}/")
    return n_windows


# ── Compute: fill in this task's slice of windows (array job, GPU) ─────────


def run_variant_chunk(
    model: AlphaGenome,
    device: str,
    row: pd.Series,
    out_dir: str,
    window_indices: list,
    max_windows: int = None,
) -> None:
    """Run the model for exactly `window_indices` (this task's chunk) and save results.

    Assumes setup_variant() has already run -- opens ref.npy/alt.npy/done_mask.npy in
    mode="r+" only (never creates them), so concurrent array tasks writing disjoint
    index ranges never race on file creation.
    """
    chrom = row["chrom"]
    window_start = int(row["window_start"])
    var_offset = int(row["var_pos_in_window"])
    anchor_offset = int(row["anchor_pos_in_window"])
    atg_offset = int(row["atg_pos_in_window"])
    ref_allele = row["ref"].upper()
    alt_allele = row["alt"].upper()

    k_values = k_values_for(row, max_windows)
    n_windows = len(k_values)

    for name in ("ref", "alt", "done_mask"):
        path = os.path.join(out_dir, f"{name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} missing -- run with --setup_only first (see script docstring)"
            )

    with Fasta(REF_FASTA) as fa:
        genomic_ref = str(fa[chrom][window_start - 1 : window_start - 1 + MODEL_LEN]).upper()

    assert len(genomic_ref) == MODEL_LEN, f"Genome extraction short: {len(genomic_ref)} bp"
    assert genomic_ref[var_offset] == ref_allele, (
        f"REF mismatch at offset {var_offset}: expected {ref_allele}, got {genomic_ref[var_offset]}"
    )
    assert genomic_ref[atg_offset : atg_offset + 3] == "ATG", (
        f"ATG not found at offset {atg_offset}: got {genomic_ref[atg_offset : atg_offset + 3]!r}"
    )
    genomic_alt = genomic_ref[:var_offset] + alt_allele + genomic_ref[var_offset + 1 :]

    ref_arr = np.lib.format.open_memmap(os.path.join(out_dir, "ref.npy"), mode="r+")
    alt_arr = np.lib.format.open_memmap(os.path.join(out_dir, "alt.npy"), mode="r+")
    done_mask = np.lib.format.open_memmap(os.path.join(out_dir, "done_mask.npy"), mode="r+")
    assert ref_arr.shape == alt_arr.shape == (n_windows, MODEL_LEN)
    assert done_mask.shape == (n_windows,)

    print(
        f"  window_indices {window_indices[0]}..{window_indices[-1]} ({len(window_indices)} windows)"
    )
    for i in tqdm(window_indices, desc="K-sweep chunk"):
        if done_mask[i]:
            continue
        k = k_values[i]

        ref_seq = build_window(genomic_ref, anchor_offset, k)
        alt_seq = build_window(genomic_alt, anchor_offset, k)

        # Batch: [ref, alt] — single forward pass (no revcomp: not saved, so not computed)
        batch = torch.cat([onehot(ref_seq), onehot(alt_seq)], dim=0).to(device)

        preds = model.predict(
            batch, organism_index=0, named_outputs=True, heads=("rna_seq",), resolutions=(1,)
        )
        rna1bp = preds.rna_seq[1]
        hepg2 = rna1bp.select(biosample_name="HepG2")
        out = hepg2.tensor.cpu().numpy().mean(axis=-1)  # (2, 131_072)

        ref_vals, alt_vals = out

        ref_arr[i] = ref_vals.astype(DTYPE)
        alt_arr[i] = alt_vals.astype(DTYPE)
        done_mask[i] = True

    ref_arr.flush()
    alt_arr.flush()
    done_mask.flush()
    print(f"  Saved {len(window_indices)} windows → {out_dir}/")


# ── Entry point ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AlphaGenome random-insertion sliding-window SNV predictions "
        "(HepG2 RNA-seq, 1 bp resolution) for LDLR promoter variants."
    )
    parser.add_argument(
        "--variants_tsv",
        default=VARIANTS_TSV,
        help=f"TSV with variant/window definitions (default: {VARIANTS_TSV})",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Only run this variant (must match the 'variant' column exactly, e.g. 'c.-121T>C'). "
        "Default: run every row in the TSV.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help=f"output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument("--device", default=None, help="cuda or cpu (auto-detected if omitted)")
    parser.add_argument(
        "--max_windows",
        type=int,
        default=None,
        help="Smoke-test hook: only run the first N windows (of ~1,639) per variant. "
        "Use a separate --out_dir for this -- output shape won't match a full run.",
    )
    parser.add_argument(
        "--setup_only",
        action="store_true",
        help="Preallocate ref.npy/alt.npy/done_mask.npy + write meta.tsv, then exit. "
        "No GPU/model needed. Run this once before launching a SLURM array of compute tasks.",
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=None,
        help="This task's index in [0, num_tasks). Default: $SLURM_ARRAY_TASK_ID, else 0.",
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=None,
        help="Total number of parallel tasks splitting the K-sweep. "
        "Default: $SLURM_ARRAY_TASK_COUNT, else 1 (i.e. run everything in this process).",
    )
    args = parser.parse_args()

    out_dir_base = args.out_dir if args.out_dir is not None else OUTPUT_DIR
    device = (
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    task_id = (
        args.task_id if args.task_id is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    )
    num_tasks = (
        args.num_tasks
        if args.num_tasks is not None
        else int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    )

    variants_df = pd.read_csv(args.variants_tsv, sep="\t")
    if args.variant is not None:
        variants_df = variants_df[variants_df["variant"] == args.variant]
        if variants_df.empty:
            raise ValueError(f"No row matching --variant {args.variant!r} in {args.variants_tsv}")

    safe_names = {"c.-121T>C": "c.-121T_C", "c.-120C>T": "c.-120C_T"}

    if args.setup_only:
        print(f"Output   : {out_dir_base}\n")
        for _, row in variants_df.iterrows():
            name = row["variant"]
            safe_name = safe_names.get(
                name, name.replace(">", "_").replace(".", "").replace("*", "")
            )
            out_dir = os.path.join(out_dir_base, safe_name)
            print(f"Variant  : {row['chrom']}:{row['var_pos']} {row['ref']}>{row['alt']}  ({name})")
            setup_variant(row, out_dir, max_windows=args.max_windows)
        print("\nSetup done. Now launch the compute array job.")
        return

    print(f"Device   : {device}")
    print(f"Task     : {task_id} / {num_tasks}")
    print(f"Output   : {out_dir_base}\n")

    model = load_model(device)

    for _, row in variants_df.iterrows():
        name = row["variant"]
        print(f"Variant  : {row['chrom']}:{row['var_pos']} {row['ref']}>{row['alt']}  ({name})")
        safe_name = safe_names.get(name, name.replace(">", "_").replace(".", "").replace("*", ""))
        out_dir = os.path.join(out_dir_base, safe_name)
        n_windows = len(k_values_for(row, args.max_windows))
        window_indices = partition_indices(n_windows, task_id, num_tasks)
        if not window_indices:
            print(f"  Task {task_id} has no windows to do for {name} (num_tasks > n_windows)")
            continue
        run_variant_chunk(
            model=model,
            device=device,
            row=row,
            out_dir=out_dir,
            window_indices=window_indices,
            max_windows=args.max_windows,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
