from alphagenome.models import dna_client
from pyfaidx import Fasta
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/8-aphagenome"
PRED_DIR = f"{DATA_DIR}/predictions"
REF_FASTA = "/scratch/st-cdeboer-1/sambina/reference_genome/hg38.fa"
os.makedirs(PRED_DIR, exist_ok=True)

# ── API ────────────────────────────────────────────────────────────────────
API_KEY = "ALPHAGENOME_API_KEY_REMOVED"
model = dna_client.create(API_KEY)

# ── Tissue / ontology terms ────────────────────────────────────────────────
# Leave [] for all tracks, or restrict to tissues of interest.
# Examples: 'UBERON:0000955' (brain), 'EFO:0002067' (K562)
ONTOLOGY_TERMS = []

# ── Resolution for saved predictions (bp) ─────────────────────────────────
BIN_SIZE = 128

# ── Load sliding windows ───────────────────────────────────────────────────
windows = {
    "C228T": pd.read_csv(f"{DATA_DIR}/tert_C228T_sliding_windows.tsv", sep="\t"),
    "C250T": pd.read_csv(f"{DATA_DIR}/tert_C250T_sliding_windows.tsv", sep="\t"),
}
for name, df in windows.items():
    print(f"{name}: {len(df)} windows, var_pos_in_window = {df['var_pos_in_window'].tolist()}")

# ── Run predictions ────────────────────────────────────────────────────────
fa = Fasta(REF_FASTA)

for variant_name, df in windows.items():
    out_dir = f"{PRED_DIR}/{variant_name}"
    os.makedirs(out_dir, exist_ok=True)

    for i, row in tqdm(df.iterrows(), total=len(df), desc=variant_name):
        ref_path = f"{out_dir}/window_{i}_ref.npy"
        alt_path = f"{out_dir}/window_{i}_alt.npy"

        if os.path.exists(ref_path) and os.path.exists(alt_path):
            print(f"  [{variant_name} win {i}] already exists, skipping")
            continue

        # TSV is 1-based; pyfaidx slicing is 0-based half-open
        start_0 = int(row["window_start"]) - 1
        seq_len = dna_client.SEQUENCE_LENGTH_1MB
        ref_seq = str(fa[row["chrom"]][start_0 : start_0 + seq_len]).upper()

        # Apply SNV at its 0-based offset within the window
        var_offset = int(row["var_pos_in_window"])
        alt_seq = ref_seq[:var_offset] + row["alt"] + ref_seq[var_offset + 1 :]

        ref_output = model.predict_sequence(
            sequence=ref_seq,
            requested_outputs=[dna_client.OutputType.RNA_SEQ],
            ontology_terms=ONTOLOGY_TERMS,
        )
        alt_output = model.predict_sequence(
            sequence=alt_seq,
            requested_outputs=[dna_client.OutputType.RNA_SEQ],
            ontology_terms=ONTOLOGY_TERMS,
        )

        np.save(ref_path, ref_output.rna_seq.change_resolution(BIN_SIZE).values)
        np.save(alt_path, alt_output.rna_seq.change_resolution(BIN_SIZE).values)
        print(
            f"  [{variant_name} win {i}] saved — shape "
            f"{ref_output.rna_seq.change_resolution(BIN_SIZE).values.shape}"
        )

fa.close()

# ── Plot 1: Per-window RNA-seq tracks ─────────────────────────────────────
ZOOM_BP = 10_000
TERT_TSS = 1_295_047  # hg38, minus strand; upstream = negative in TSS-relative coords

for variant_name, df in windows.items():
    out_dir = f"{PRED_DIR}/{variant_name}"
    df_sorted = df.sort_values("var_pos_in_window")
    n_win = len(df_sorted)
    var_pos = int(df_sorted["var_pos"].iloc[0])
    x_min, x_max = var_pos - ZOOM_BP, var_pos + ZOOM_BP
    # TSS-relative coords: TSS - genomic_pos (minus strand: upstream < 0, downstream > 0)
    x_min_tss = TERT_TSS - x_max
    x_max_tss = TERT_TSS - x_min
    var_pos_tss = TERT_TSS - var_pos

    fig, axes = plt.subplots(n_win, 1, figsize=(14, 1.6 * n_win), sharex=True)
    if n_win == 1:
        axes = [axes]

    for ax, (orig_idx, row) in zip(axes, df_sorted.iterrows()):
        ref_path = f"{out_dir}/window_{orig_idx}_ref.npy"
        alt_path = f"{out_dir}/window_{orig_idx}_alt.npy"

        if not (os.path.exists(ref_path) and os.path.exists(alt_path)):
            ax.set_visible(False)
            continue

        ref_vals = np.load(ref_path)  # (n_bins, n_tracks)
        alt_vals = np.load(alt_path)

        start_0 = int(row["window_start"]) - 1
        bin_pos = start_0 + np.arange(ref_vals.shape[0]) * BIN_SIZE

        mask = (bin_pos >= x_min) & (bin_pos <= x_max)
        x = TERT_TSS - bin_pos[mask]
        ref_mean = ref_vals[mask].mean(axis=1)
        alt_mean = alt_vals[mask].mean(axis=1)

        ax.fill_between(x, ref_mean, alpha=0.45, color="steelblue", label="REF")
        ax.fill_between(x, alt_mean, alpha=0.45, color="crimson", label="ALT")
        ax.plot(x, ref_mean, color="steelblue", lw=0.7)
        ax.plot(x, alt_mean, color="crimson", lw=0.7, linestyle="--")
        ax.axvline(var_pos_tss, color="black", lw=1, linestyle=":", alpha=0.7)
        ax.axvline(0, color="dimgrey", lw=0.8, linestyle="--", alpha=0.5)  # TSS

        vp = int(row["var_pos_in_window"])
        ax.set_ylabel(f"{vp} bp", fontsize=8, rotation=0, ha="right", va="center", labelpad=4)
        ax.set_yticks([])
        ax.set_xlim(x_min_tss, x_max_tss)
        ax.spines[["top", "right", "left"]].set_visible(False)

    axes[0].legend(
        fontsize=8,
        loc="upper right",
        framealpha=0.8,
        handles=[
            plt.matplotlib.patches.Patch(color="steelblue", label="REF"),
            plt.matplotlib.patches.Patch(color="crimson", label="ALT"),
        ],
    )
    axes[-1].set_xlabel("Distance from TSS (bp)", fontsize=10)

    fig.text(0.0, 0.5, "var_pos_in_window →", va="center", rotation=90, fontsize=9, color="dimgrey")
    fig.suptitle(
        f"TERT {variant_name} — AlphaGenome RNA-seq tracks  "
        f"({BIN_SIZE} bp res, ±{ZOOM_BP // 1000} kb around variant)",
        fontsize=12,
    )
    plt.tight_layout()
    out_pdf = f"{DATA_DIR}/tert_{variant_name}_window_tracks.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_pdf}")

# ── Plot 2: Expression at variant bin vs window position ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, (variant_name, df) in zip(axes, windows.items()):
    out_dir = f"{PRED_DIR}/{variant_name}"
    records = []

    for i, row in df.iterrows():
        ref_path = f"{out_dir}/window_{i}_ref.npy"
        alt_path = f"{out_dir}/window_{i}_alt.npy"
        if not (os.path.exists(ref_path) and os.path.exists(alt_path)):
            continue

        ref_vals = np.load(ref_path)
        alt_vals = np.load(alt_path)

        var_bin = int(row["var_pos_in_window"]) // BIN_SIZE
        records.append(
            {
                "var_pos_in_window": row["var_pos_in_window"],
                "ref_expr": float(ref_vals[var_bin].mean()),
                "alt_expr": float(alt_vals[var_bin].mean()),
            }
        )

    if not records:
        ax.set_title(f"{variant_name} — no predictions yet")
        continue

    res = pd.DataFrame(records).sort_values("var_pos_in_window")

    ax.plot(
        res["var_pos_in_window"], res["ref_expr"], "o-", color="dimgrey", label="REF", lw=2, ms=7
    )
    ax.plot(
        res["var_pos_in_window"], res["alt_expr"], "o-", color="crimson", label="ALT", lw=2, ms=7
    )

    ax.set_xlabel("Variant position in window (bp from left edge)", fontsize=11)
    ax.set_ylabel("Mean predicted RNA-seq at variant bin", fontsize=11)
    ax.set_title(f"TERT {variant_name}", fontsize=12)
    ax.legend()

plt.suptitle("Predicted expression at variant bin vs window position", fontsize=13)
plt.tight_layout()
out_pdf = f"{DATA_DIR}/tert_alphagenome_score_vs_position.pdf"
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()
print(f"Saved {out_pdf}")
