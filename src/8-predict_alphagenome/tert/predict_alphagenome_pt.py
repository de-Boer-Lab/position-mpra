"""
Predict AlphaGenome RNA-seq (HepG2, 128 bp resolution) for a single
SNV using the PyTorch model, with a sliding-window design.

Model input / output
--------------------
  Input  : 131,072 bp  (2^17)
  Output : 1,024 bins at 128 bp resolution  →  covers all 131,072 bp
  Track  : single HepG2 RNA-seq track  →  (1024,) per window

Sliding window design
---------------------
  1,024 windows — one per output bin.
  Window b places the variant at the start of output bin b:

    [ n_random_left = b × 128 bp  random (shared prefix of RAND_LEFT) ]
    [ left_real bp of real genomic, constant per variant               ]
    [ VARIANT (1 bp)                                                   ]
    [ right_real = 131,072 − n_random_left − left_real − 1  real bp   ]

  As b increases: left random grows, right real context shrinks.

Saved per window — {out_dir}/bin_{b}.json:
  ref              (1024,) float32  — forward strand, reference allele
  alt              (1024,) float32  — forward strand, alternate allele
  ve               (1024,) float32  —  ref - alt (forward)
  ref_rc           (1024,) float32  — revcomp ref, flipped to forward orientation
  alt_rc           (1024,) float32  — revcomp alt, flipped to forward orientation
  ve_rc            (1024,) float32  —  ref_rc - alt_rc
  var_bin          scalar int64     — forward output bin of variant (= b)
  var_pos_in_model scalar int64     — bp position of variant in 131,072 bp input
  n_random_left    scalar int64     — random bp on the left
  right_real       scalar int64     — real genomic bp to the right of variant

Note: ref_rc/alt_rc/ve_rc are saved with [::-1] applied, so bin i in revcomp
      aligns with the same genomic window as bin i in forward.

Usage
-----
  python predict_alphagenome_pt.py \\
      --chrom chr5 --var_pos 1295135 --ref G --alt A --name C250T

  Optional flags:
    --window_start INT   left genomic anchor, 1-based (default: var_pos − 68)
    --out_dir PATH       output directory (default: OUTPUT_DIR/{name})
    --device cpu|cuda    override device detection
"""

import argparse
import json
import os

import numpy as np
import torch
from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.named_outputs import TrackMetadataCatalog
from pyfaidx import Fasta
from tqdm import tqdm

# ── Fixed paths / constants ────────────────────────────────────────────────
DATA_DIR = "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/8-aphagenome"
REF_FASTA = "/scratch/st-cdeboer-1/sambina/reference_genome/hg38.fa"
WEIGHTS = f"{DATA_DIR}/pytorch/model_all_folds.safetensors"
OUTPUT_DIR = f"{DATA_DIR}/pytorch/predictions"

MODEL_LEN = 131_072  # model input length (bp)
N_BINS = MODEL_LEN // 128  # = 1,024 output bins

# Shared left-side random string: every window uses a prefix of this.
_MAX_RAND = (N_BINS - 1) * 128  # = 130,944 bp
_BASES = np.array(list("ACGT"))
RAND_LEFT = "".join(_BASES[np.random.default_rng(seed=42).integers(0, 4, size=_MAX_RAND)])

# Reverse-complement lookup
_COMP = str.maketrans("ACGT", "TGCA")

# One-hot encoding lookup table
_LUT = np.zeros(256, dtype=np.int8)
for _i, _b in enumerate("ACGT"):
    _LUT[ord(_b)] = _i


# ── Sequence utilities ─────────────────────────────────────────────────────


def revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def onehot(seq: str) -> torch.Tensor:
    """DNA string → (1, L, 4) float32 NLC tensor  (A=0 C=1 G=2 T=3)."""
    arr = _LUT[np.frombuffer(seq.encode(), dtype=np.uint8)]
    return torch.from_numpy(np.eye(4, dtype=np.float32)[arr]).unsqueeze(0)


def build_window(g_left: str, allele: str, g_right: str, n_random_left: int) -> str:
    """Assemble one 131,072 bp input sequence."""
    seq = RAND_LEFT[:n_random_left] + g_left + allele + g_right
    assert len(seq) == MODEL_LEN, f"sequence length {len(seq)} != {MODEL_LEN}"
    return seq


# ── Model loading ──────────────────────────────────────────────────────────


def load_model(device: str) -> AlphaGenome:
    model = AlphaGenome.from_pretrained(WEIGHTS, device=device)
    catalog = TrackMetadataCatalog.load_builtin(organism=0)
    model.set_track_metadata_catalog(catalog)
    model.eval()
    return model


# ── Core prediction loop ───────────────────────────────────────────────────


def run_predictions(
    model: AlphaGenome,
    device: str,
    chrom: str,
    var_pos: int,
    ref_allele: str,
    alt_allele: str,
    window_start: int,
    out_dir: str,
) -> None:
    """Run all 1,024 sliding windows for one variant and save results."""
    os.makedirs(out_dir, exist_ok=True)

    left_real = var_pos - window_start  # real genomic bp to the left of variant

    with Fasta(REF_FASTA) as fa:
        genomic = str(fa[chrom][window_start - 1 : window_start - 1 + MODEL_LEN]).upper()

    assert len(genomic) == MODEL_LEN, f"Genome extraction short: {len(genomic)} bp"
    assert genomic[left_real].upper() == ref_allele.upper(), (
        f"REF mismatch at offset {left_real}: expected {ref_allele}, got {genomic[left_real]}"
    )

    g_left = genomic[:left_real]
    g_right_full = genomic[left_real + 1 :]  # max_right_real bp

    for b in tqdm(range(N_BINS), desc=f"bin"):
        out_path = os.path.join(out_dir, f"bin_{b}.json")
        if os.path.exists(out_path):
            continue

        n_random_left = b * 128
        right_real = MODEL_LEN - n_random_left - left_real - 1
        g_right = g_right_full[:right_real]

        ref_seq = build_window(g_left, ref_allele, g_right, n_random_left)
        alt_seq = build_window(g_left, alt_allele, g_right, n_random_left)

        # Batch: [ref, alt, ref_rc, alt_rc] — single forward pass
        batch = torch.cat(
            [onehot(ref_seq), onehot(alt_seq), onehot(revcomp(ref_seq)), onehot(revcomp(alt_seq))],
            dim=0,
        ).to(device)

        preds = model.predict(batch, organism_index=0, named_outputs=True, heads=("rna_seq",))
        rna128 = preds.rna_seq[128]
        hepg2 = rna128.select(biosample_name="HepG2")
        out = hepg2.tensor.cpu().numpy().mean(axis=-1)  # (4, 1024)

        ref_vals, alt_vals, ref_rc_vals, alt_rc_vals = out
        ve_vals = ref_vals - alt_vals
        # flip revcomp so bin i aligns with the same genomic window as forward bin i
        ref_rc_vals = ref_rc_vals[::-1]
        alt_rc_vals = alt_rc_vals[::-1]
        ve_rc_vals = ref_rc_vals - alt_rc_vals

        with open(out_path, "w") as f:
            json.dump(
                {
                    "ref": ref_vals.tolist(),
                    "alt": alt_vals.tolist(),
                    "ve": ve_vals.tolist(),
                    "ref_rc": ref_rc_vals.tolist(),
                    "alt_rc": alt_rc_vals.tolist(),
                    "ve_rc": ve_rc_vals.tolist(),
                    "var_bin": b,
                    "var_pos_in_model": n_random_left + left_real,
                    "n_random_left": n_random_left,
                    "right_real": right_real,
                },
                f,
            )


# ── Entry point ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AlphaGenome sliding-window SNV predictions (HepG2 RNA-seq, 128 bp)."
    )
    parser.add_argument("--chrom", required=True, help="chromosome, e.g. chr5")
    parser.add_argument("--var_pos", required=True, type=int, help="variant position, 1-based hg38")
    parser.add_argument("--ref", required=True, help="reference allele (single base)")
    parser.add_argument("--alt", required=True, help="alternate allele (single base)")
    parser.add_argument("--name", required=True, help="label for output directory, e.g. C250T")
    parser.add_argument(
        "--window_start",
        type=int,
        default=None,
        help="left genomic anchor, 1-based (default: var_pos − 68)",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help=f"output directory (default: {OUTPUT_DIR}/<name>)",
    )
    parser.add_argument("--device", default=None, help="cuda or cpu (auto-detected if omitted)")
    args = parser.parse_args()

    ref_allele = args.ref.upper()
    alt_allele = args.alt.upper()
    window_start = args.window_start if args.window_start is not None else args.var_pos - 68
    out_dir = args.out_dir if args.out_dir is not None else os.path.join(OUTPUT_DIR, args.name)
    device = (
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Variant  : {args.chrom}:{args.var_pos} {ref_allele}>{alt_allele}  ({args.name})")
    print(
        f"Window   : {args.chrom}:{window_start}–{window_start + MODEL_LEN - 1}  ({MODEL_LEN:,} bp)"
    )
    print(f"Device   : {device}")
    print(f"Output   : {out_dir}\n")

    model = load_model(device)

    run_predictions(
        model=model,
        device=device,
        chrom=args.chrom,
        var_pos=args.var_pos,
        ref_allele=ref_allele,
        alt_allele=alt_allele,
        window_start=window_start,
        out_dir=out_dir,
    )

    print(f"\nDone — {N_BINS} bins saved to {out_dir}/")


if __name__ == "__main__":
    main()
