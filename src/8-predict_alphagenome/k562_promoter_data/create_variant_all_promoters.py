"""
AlphaGenome predictions (1 bp resolution, HepG2 RNA-seq) for a synthetic -150
promoter variant, with random-DNA insertions at -200 (upstream) and -100
(downstream) of the TSS, across all 17,547 K562-open-chromatin canonical-TSS
genes whose exon2 fits inside the model's downstream window reach.

Experiment design
------------------
For every gene:
  - A fixed-transition substitution (A<->G, C<->T) is introduced at TSS-150.
  - Three conditions are predicted (ref + alt each), all sharing the same
    -150 variant:
      baseline    no insertion
      upstream    K random bp inserted at TSS-200
      downstream  K random bp inserted at TSS-100
  - K (the insertion length) is one of {25, 50, 75, 100}, assigned once per
    gene so the four length categories are balanced (~equal gene counts).
  - K random bp are inserted at the anchor point and K bp are trimmed off the
    far upstream edge of the window (52,429 bp away from TSS -- harmless),
    keeping the window fixed at MODEL_LEN and leaving TSS/exon1/exon2 at a
    FIXED offset in every condition (same trick as predict_alphagenome_ldlr.py).
    Note: for the downstream (-100) insertion, the -150 variant itself sits
    upstream of the insertion anchor and so shifts left by K bp within the
    window -- this doesn't matter for analysis since only exon2 (which never
    moves) is summarized downstream.
  - Random insertion content: each gene draws its OWN random K-bp sequence
    (seeded per-gene, so different genes don't share identical inserted
    sequence -- avoids one unlucky random string, e.g. one that happens to
    create/destroy a motif, systematically biasing every gene). The SAME
    per-gene sequence is reused at both -200 and -100 for that gene, so the
    upstream-vs-downstream comparison isolates the effect of insertion
    POSITION, not a confound from different inserted content.

Sequences are always built in TRANSCRIPT orientation (reverse-complemented
for minus-strand genes before any offset math), so TSS sits at a fixed
offset (TSS_OFFSET = UPSTREAM_LEN = 52,429) regardless of genomic strand.

Inputs (already built earlier in this project):
  k562_tss_in_open_chromatin.canonical_gene.bed   -- one canonical TSS per gene
  k562_exon1_exon2.canonical_gene.bed             -- exon1/exon2 coords per gene

Saved to {OUTPUT_DIR}/predictions/:
  promoters_metadata.tsv
      one row per included gene: gene, transcript_id, chrom, strand,
      window_start_genomic, tss_offset, variant_offset, ref, alt,
      upstream_ins_offset, downstream_ins_offset, length_category,
      exon2_start_offset, exon2_end_offset
  {condition}_ref.npy, {condition}_alt.npy   for condition in baseline/upstream/downstream
      memory-mapped (n_genes, 131_072) float16 arrays, HepG2 track, forward strand only.
  done_mask.npy   (n_genes, 3) bool memmap -- columns = [baseline, upstream, downstream]

Two-phase, SLURM-array-safe (same race-avoidance pattern as predict_alphagenome_ldlr.py):
  1. Setup (once, no GPU): builds metadata + preallocates files.
       python create_variant_all_promoters.py --setup_only
  2. Compute (array job, GPU): each task only opens files in mode="r+".
       python create_variant_all_promoters.py --task_id $SLURM_ARRAY_TASK_ID --num_tasks 100

Usage
-----
  python create_variant_all_promoters.py --setup_only
  python create_variant_all_promoters.py --task_id 0 --num_tasks 100 --device cuda
  python create_variant_all_promoters.py --max_genes 5 --device cpu   # smoke test
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
PROMOTER_DIR = "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/8-aphagenome/all_k562_promoters"
REF_FASTA = "/scratch/st-cdeboer-1/sambina/reference_genome/hg38.fa"
CHROM_SIZES = "/scratch/st-cdeboer-1/sambina/reference_genome/hg38.chrom.sizes"
WEIGHTS = "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/8-aphagenome/pytorch/model_all_folds.safetensors"
OUTPUT_DIR = f"{PROMOTER_DIR}/predictions"

TSS_BED = f"{PROMOTER_DIR}/k562_tss_in_open_chromatin.canonical_gene.bed"
EXON_BED = f"{PROMOTER_DIR}/k562_exon1_exon2.canonical_gene.bed"
METADATA_TSV = f"{OUTPUT_DIR}/promoters_metadata.tsv"

MODEL_LEN = 131_072  # model input length (bp), fixed by the model architecture
UPSTREAM_LEN = 52_429  # bp upstream of TSS in the window (matches predict_alphagenome_ldlr.py)
DOWNSTREAM_LEN = MODEL_LEN - UPSTREAM_LEN  # = 78,643 bp downstream of TSS
TSS_OFFSET = UPSTREAM_LEN  # TSS sits at this offset in every transcript-oriented window

VARIANT_REL_POS = 150  # TSS - 150
UPSTREAM_INS_REL_POS = 200  # TSS - 200
DOWNSTREAM_INS_REL_POS = 100  # TSS - 100
LENGTH_CATEGORIES = [25, 50, 75, 100]
CONDITIONS = ["baseline", "upstream", "downstream"]

DTYPE = np.float16
RNG_SEED = 42

TRANSITION = {"A": "G", "G": "A", "C": "T", "T": "C"}
COMP = str.maketrans("ACGT", "TGCA")

# One-hot encoding lookup table
_LUT = np.zeros(256, dtype=np.int8)
for _i, _b in enumerate("ACGT"):
    _LUT[ord(_b)] = _i


def revcomp(seq: str) -> str:
    return seq.translate(COMP)[::-1]


def onehot(seq: str) -> torch.Tensor:
    """DNA string → (1, L, 4) float32 NLC tensor  (A=0 C=1 G=2 T=3)."""
    arr = _LUT[np.frombuffer(seq.encode(), dtype=np.uint8)]
    return torch.from_numpy(np.eye(4, dtype=np.float32)[arr]).unsqueeze(0)


# ── Metadata construction (no GPU) ──────────────────────────────────────────


def load_tss_and_exons() -> pd.DataFrame:
    tss_rows = []
    with open(TSS_BED) as f:
        for line in f:
            chrom, start, end, name, score, strand = line.rstrip("\n").split("\t")
            gene, tx, tag = name.split("|")
            tss_rows.append(
                {
                    "gene": gene,
                    "transcript_id": tx,
                    "chrom": chrom,
                    "strand": strand,
                    "tss": int(start) + 1,
                }
            )  # bed start is 0-based -> +1 for 1-based TSS position
    tss_df = pd.DataFrame(tss_rows).set_index("transcript_id")

    exon2 = {}
    with open(EXON_BED) as f:
        for line in f:
            chrom, start, end, name, score, strand = line.rstrip("\n").split("\t")
            gene, tx, exon = name.split("|")
            if exon == "exon2":
                exon2[tx] = (int(start) + 1, int(end))  # 1-based inclusive [start, end]

    tss_df["exon2_start_1based"] = tss_df.index.map(lambda tx: exon2.get(tx, (None, None))[0])
    tss_df["exon2_end_1based"] = tss_df.index.map(lambda tx: exon2.get(tx, (None, None))[1])
    return tss_df.reset_index()


def build_metadata() -> pd.DataFrame:
    df = load_tss_and_exons()
    df = df.dropna(subset=["exon2_start_1based", "exon2_end_1based"]).copy()
    df["exon2_start_1based"] = df["exon2_start_1based"].astype(int)
    df["exon2_end_1based"] = df["exon2_end_1based"].astype(int)

    # window_start_genomic: 1-based genomic (plus-strand) coordinate of the
    # window's lowest-coordinate base, before any strand-orientation flip.
    is_plus = df["strand"] == "+"
    df["window_start_genomic"] = np.where(
        is_plus, df["tss"] - UPSTREAM_LEN, df["tss"] - DOWNSTREAM_LEN + 1
    )
    window_end_genomic = df["window_start_genomic"] + MODEL_LEN - 1

    # Chromosome-boundary filter: window must fit entirely within the chromosome
    # (genes near a telomere/contig end don't have enough flanking sequence).
    chrom_sizes = {}
    with open(CHROM_SIZES) as f:
        for line in f:
            chrom, size = line.split()
            chrom_sizes[chrom] = int(size)
    chrom_len = df["chrom"].map(chrom_sizes)
    in_bounds = (df["window_start_genomic"] >= 1) & (window_end_genomic <= chrom_len)
    n_oob = (~in_bounds).sum()
    df = df[in_bounds].reset_index(drop=True)
    window_end_genomic = window_end_genomic[in_bounds].reset_index(drop=True)
    is_plus = is_plus[in_bounds].reset_index(drop=True)
    print(
        f"Chromosome-boundary filter: dropped {n_oob} genes whose window runs off the chromosome; {len(df)} remain"
    )

    # Transcript-oriented offset of a 1-based genomic position within the window.
    def offset(pos, strand_is_plus, w_start, w_end):
        return np.where(strand_is_plus, pos - w_start, w_end - pos)

    df["exon2_start_offset"] = offset(
        df["exon2_start_1based"], is_plus, df["window_start_genomic"], window_end_genomic
    )
    df["exon2_end_offset"] = offset(
        df["exon2_end_1based"], is_plus, df["window_start_genomic"], window_end_genomic
    )
    # exon2_start_offset > exon2_end_offset on minus strand (transcript order flips genomic order) -- normalize
    lo = np.minimum(df["exon2_start_offset"], df["exon2_end_offset"])
    hi = np.maximum(df["exon2_start_offset"], df["exon2_end_offset"])
    df["exon2_start_offset"], df["exon2_end_offset"] = lo, hi + 1  # +1 -> half-open, for slicing

    # Feasibility filter: exon2 must fit fully inside the window's downstream reach.
    feasible = df["exon2_end_offset"] <= MODEL_LEN
    n_dropped = (~feasible).sum()
    df = df[feasible].reset_index(drop=True)
    print(
        f"Feasibility filter: dropped {n_dropped} genes with exon2 beyond window reach; {len(df)} remain"
    )

    # Fixed offsets, same for every gene by construction (TSS always at TSS_OFFSET).
    df["tss_offset"] = TSS_OFFSET
    df["variant_offset"] = TSS_OFFSET - VARIANT_REL_POS
    df["upstream_ins_offset"] = TSS_OFFSET - UPSTREAM_INS_REL_POS
    df["downstream_ins_offset"] = TSS_OFFSET - DOWNSTREAM_INS_REL_POS

    # Balanced random assignment to the 4 length categories.
    rng = np.random.default_rng(RNG_SEED)
    n = len(df)
    cats = np.resize(LENGTH_CATEGORIES, n)
    # pad to exact length then trim, resize already tiles evenly; shuffle for randomness
    idx = rng.permutation(n)
    cats_shuffled = np.empty(n, dtype=int)
    cats_shuffled[idx] = cats
    df["length_category"] = cats_shuffled

    # Per-gene random insertion sequence (own draw per gene, reused for both
    # -200 and -100 in that gene so upstream/downstream differ only in
    # position, not content). Drawn in fixed row order for reproducibility.
    df["insertion_seq"] = [
        "".join(np.array(list("ACGT"))[rng.integers(0, 4, size=int(k))])
        for k in df["length_category"]
    ]

    # Reference/alt allele via fixed transition rule -- filled in during setup
    # once the reference genome is available (needs the actual ref base at
    # variant_offset, which requires pulling the sequence).
    return df.reset_index(drop=True)


def fetch_ref_and_alt(fa: Fasta, row: pd.Series) -> tuple:
    """Pull the transcript-oriented MODEL_LEN window and the variant's ref/alt bases."""
    w_start = int(row["window_start_genomic"])
    seq = str(fa[row["chrom"]][w_start - 1 : w_start - 1 + MODEL_LEN]).upper()
    assert len(seq) == MODEL_LEN, f"Genome extraction short: {len(seq)} bp for {row['gene']}"
    if row["strand"] == "-":
        seq = revcomp(seq)
    ref_base = seq[int(row["variant_offset"])]
    alt_base = TRANSITION[ref_base]
    return seq, ref_base, alt_base


# ── Sequence construction for the 3 conditions ──────────────────────────────


def build_variant_seq(seq: str, variant_offset: int, allele: str) -> str:
    return seq[:variant_offset] + allele + seq[variant_offset + 1 :]


def build_insertion_seq(seq: str, insertion_offset: int, k: int, insertion_seq: str) -> str:
    """Trim k bp off the far-upstream edge (index 0), insert k random bp right
    after insertion_offset. Total length stays MODEL_LEN; everything from
    insertion_offset+1 onward (incl. TSS/exon1/exon2) keeps its original offset.
    """
    if k == 0:
        return seq
    assert len(insertion_seq) == k, f"insertion_seq length {len(insertion_seq)} != k={k}"
    left_kept = seq[k : insertion_offset + 1]
    right_unchanged = seq[insertion_offset + 1 :]
    out = left_kept + insertion_seq + right_unchanged
    assert len(out) == MODEL_LEN, f"sequence length {len(out)} != {MODEL_LEN}"
    return out


def sequences_for_condition(seq_ref: str, seq_alt: str, condition: str, row: pd.Series) -> tuple:
    if condition == "baseline":
        return seq_ref, seq_alt
    k = int(row["length_category"])
    insertion_seq = row["insertion_seq"]
    anchor = (
        int(row["upstream_ins_offset"])
        if condition == "upstream"
        else int(row["downstream_ins_offset"])
    )
    return (
        build_insertion_seq(seq_ref, anchor, k, insertion_seq),
        build_insertion_seq(seq_alt, anchor, k, insertion_seq),
    )


# ── Model loading ──────────────────────────────────────────────────────────


def load_model(device: str) -> AlphaGenome:
    model = AlphaGenome.from_pretrained(WEIGHTS, device=device)
    catalog = TrackMetadataCatalog.load_builtin(organism=0)
    model.set_track_metadata_catalog(catalog)
    model.eval()
    return model


# ── Partitioning across a SLURM array ───────────────────────────────────────


def partition_indices(n: int, task_id: int, num_tasks: int) -> list:
    assert 0 <= task_id < num_tasks, f"task_id {task_id} out of range [0, {num_tasks})"
    chunk = -(-n // num_tasks)  # ceil division
    start = min(task_id * chunk, n)
    end = min(start + chunk, n)
    return list(range(start, end))


# ── Setup: build metadata + preallocate output files (run once, no GPU) ────


def setup(max_genes: int = None, out_dir: str = None) -> pd.DataFrame:
    """Build/reuse the (always-full) metadata, then preallocate array files
    for max_genes (or all genes) under out_dir. out_dir defaults to
    OUTPUT_DIR; pass a separate scratch dir for a smoke test so it doesn't
    create a shape mismatch against a real full run's array files.
    """
    out_dir = out_dir or OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(METADATA_TSV):
        df = pd.read_csv(METADATA_TSV, sep="\t")
        print(f"Reusing existing metadata: {METADATA_TSV} ({len(df)} genes)")
    else:
        df = build_metadata()
        with Fasta(REF_FASTA) as fa:
            refs, alts = [], []
            for _, row in tqdm(df.iterrows(), total=len(df), desc="fetching ref/alt alleles"):
                _, ref_base, alt_base = fetch_ref_and_alt(fa, row)
                refs.append(ref_base)
                alts.append(alt_base)
        df["ref"] = refs
        df["alt"] = alts
        df.to_csv(METADATA_TSV, sep="\t", index=False)
        print(f"Wrote metadata: {METADATA_TSV} ({len(df)} genes)")
        print(df["length_category"].value_counts().sort_index())

    if max_genes is not None:
        df = df.iloc[:max_genes].reset_index(drop=True)

    n = len(df)
    for condition in CONDITIONS:
        for name in ("ref", "alt"):
            path = os.path.join(out_dir, f"{condition}_{name}.npy")
            if os.path.exists(path):
                arr = np.lib.format.open_memmap(path, mode="r+")
                assert arr.shape == (n, MODEL_LEN), (
                    f"{path} shape mismatch, delete {out_dir} to restart"
                )
            else:
                np.lib.format.open_memmap(path, mode="w+", dtype=DTYPE, shape=(n, MODEL_LEN))

    done_mask_path = os.path.join(out_dir, "done_mask.npy")
    if os.path.exists(done_mask_path):
        done_mask = np.lib.format.open_memmap(done_mask_path, mode="r+")
        assert done_mask.shape == (n, len(CONDITIONS))
    else:
        done_mask = np.lib.format.open_memmap(
            done_mask_path, mode="w+", dtype=bool, shape=(n, len(CONDITIONS))
        )
        done_mask[:] = False

    print(f"Setup done: {n} genes x {len(CONDITIONS)} conditions, files in {out_dir}/")
    return df


# ── Compute: fill in this task's slice of genes (array job, GPU) ───────────


def run_gene_chunk(
    model: AlphaGenome, device: str, df: pd.DataFrame, gene_indices: list, out_dir: str = None
) -> None:
    out_dir = out_dir or OUTPUT_DIR
    arrays = {}
    for condition in CONDITIONS:
        for name in ("ref", "alt"):
            path = os.path.join(out_dir, f"{condition}_{name}.npy")
            arrays[(condition, name)] = np.lib.format.open_memmap(path, mode="r+")
    done_mask = np.lib.format.open_memmap(os.path.join(out_dir, "done_mask.npy"), mode="r+")

    with Fasta(REF_FASTA) as fa:
        for gi in tqdm(gene_indices, desc="genes"):
            if done_mask[gi].all():
                continue
            row = df.iloc[gi]
            seq_ref, ref_base, alt_base = fetch_ref_and_alt(fa, row)
            assert ref_base == row["ref"], (
                f"REF mismatch for {row['gene']}: expected {row['ref']}, got {ref_base}"
            )
            seq_alt = build_variant_seq(seq_ref, int(row["variant_offset"]), alt_base)

            for ci, condition in enumerate(CONDITIONS):
                if done_mask[gi, ci]:
                    continue
                cond_ref, cond_alt = sequences_for_condition(seq_ref, seq_alt, condition, row)

                batch = torch.cat([onehot(cond_ref), onehot(cond_alt)], dim=0).to(device)
                preds = model.predict(
                    batch,
                    organism_index=0,
                    named_outputs=True,
                    heads=("rna_seq",),
                    resolutions=(1,),
                )
                rna1bp = preds.rna_seq[1]
                hepg2 = rna1bp.select(biosample_name="HepG2")
                out = hepg2.tensor.cpu().numpy().mean(axis=-1)  # (2, 131_072)
                ref_vals, alt_vals = out

                arrays[(condition, "ref")][gi] = ref_vals.astype(DTYPE)
                arrays[(condition, "alt")][gi] = alt_vals.astype(DTYPE)
                done_mask[gi, ci] = True

    for arr in arrays.values():
        arr.flush()
    done_mask.flush()
    print(f"Saved {len(gene_indices)} genes -> {out_dir}/")


# ── Entry point ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AlphaGenome predictions for a synthetic -150 promoter variant "
        "+/- random-DNA insertions at -200/-100, across all K562-open-chromatin genes."
    )
    parser.add_argument("--device", default=None, help="cuda or cpu (auto-detected if omitted)")
    parser.add_argument(
        "--max_genes", type=int, default=None, help="Smoke-test hook: only run the first N genes."
    )
    parser.add_argument(
        "--setup_only", action="store_true", help="Build metadata + preallocate files, then exit."
    )
    parser.add_argument(
        "--task_id", type=int, default=None, help="Default: $SLURM_ARRAY_TASK_ID, else 0."
    )
    parser.add_argument(
        "--num_tasks", type=int, default=None, help="Default: $SLURM_ARRAY_TASK_COUNT, else 1."
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Override output dir for the array files (default: OUTPUT_DIR). "
        "Use a separate scratch dir for --max_genes smoke tests to avoid a shape "
        "mismatch against a real full run's array files.",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    task_id = (
        args.task_id if args.task_id is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    )
    num_tasks = (
        args.num_tasks
        if args.num_tasks is not None
        else int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    )

    out_dir = args.out_dir or OUTPUT_DIR

    if args.setup_only:
        setup(max_genes=args.max_genes, out_dir=out_dir)
        print("\nSetup done. Now launch the compute array job.")
        return

    df = setup(
        max_genes=args.max_genes, out_dir=out_dir
    )  # idempotent; also reloads metadata if it exists
    n = len(df)
    gene_indices = partition_indices(n, task_id, num_tasks)
    if not gene_indices:
        print(f"Task {task_id} has no genes to do (num_tasks > n_genes)")
        return

    print(f"Device   : {device}")
    print(f"Task     : {task_id} / {num_tasks}  ({len(gene_indices)} genes)")
    print(f"Output   : {out_dir}\n")

    model = load_model(device)
    run_gene_chunk(model, device, df, gene_indices, out_dir=out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
