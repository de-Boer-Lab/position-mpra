from alphagenome.data import genome
from alphagenome.models import dna_client
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/8-aphagenome"
PRED_DIR = f"{DATA_DIR}/predictions"
os.makedirs(PRED_DIR, exist_ok=True)

# ── API ────────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.environ["ALPHAGENOME_API_KEY"]
model = dna_client.create(API_KEY)

# ── Tissue / ontology terms ────────────────────────────────────────────────
ONTOLOGY_TERMS = []

# ── Resolution for saved predictions (bp) ─────────────────────────────────
BIN_SIZE = 1

# ── Load sliding windows ───────────────────────────────────────────────────
windows = {
    "C228T": pd.read_csv(f"{DATA_DIR}/tert_C228T_sliding_windows.tsv", sep="\t"),
    "C250T": pd.read_csv(f"{DATA_DIR}/tert_C250T_sliding_windows.tsv", sep="\t"),
}
for name, df in windows.items():
    print(f"{name}: {len(df)} windows, var_pos_in_window = {df['var_pos_in_window'].tolist()}")

# ── Run predictions ────────────────────────────────────────────────────────
for variant_name, df in windows.items():
    out_dir = f"{PRED_DIR}/{variant_name}"
    os.makedirs(out_dir, exist_ok=True)

    for i, row in tqdm(df.iterrows(), total=len(df), desc=variant_name):
        ref_path = f"{out_dir}/window_{i}_ref.npy"
        alt_path = f"{out_dir}/window_{i}_alt.npy"

        if os.path.exists(ref_path) and os.path.exists(alt_path):
            print(f"  [{variant_name} win {i}] already exists, skipping")
            continue

        # TSV is 1-based; genome.Interval uses 0-based half-open [start, end)
        start_0 = int(row["window_start"]) - 1
        interval = genome.Interval(
            chromosome=row["chrom"],
            start=start_0,
            end=start_0 + dna_client.SEQUENCE_LENGTH_1MB,
        )

        # genome.Variant.position is 1-based (VCF convention)
        variant = genome.Variant(
            chromosome=row["chrom"],
            position=int(row["var_pos"]),
            reference_bases=row["ref"],
            alternate_bases=row["alt"],
        )

        output = model.predict_variant(
            interval=interval,
            variant=variant,
            requested_outputs=[dna_client.OutputType.RNA_SEQ],
            ontology_terms=ONTOLOGY_TERMS,
        )

        ref_vals = output.reference.rna_seq.change_resolution(BIN_SIZE).values
        alt_vals = output.alternate.rna_seq.change_resolution(BIN_SIZE).values

        np.save(ref_path, ref_vals)
        np.save(alt_path, alt_vals)
        print(f"  [{variant_name} win {i}] saved — shape {ref_vals.shape}")

# ── Plot 1: Per-window RNA-seq tracks ─────────────────────────────────────
ZOOM_BP = 10_000  # ± bp around variant shown in each track

for variant_name, df in windows.items():
    out_dir = f"{PRED_DIR}/{variant_name}"
    df_sorted = df.sort_values("var_pos_in_window")
    n_win = len(df_sorted)
    var_pos = int(df_sorted["var_pos"].iloc[0])
    x_min, x_max = var_pos - ZOOM_BP, var_pos + ZOOM_BP

    fig, axes = plt.subplots(n_win, 1, figsize=(14, 1.6 * n_win), sharex=True)
    if n_win == 1:
        axes = [axes]

    for ax, (orig_idx, row) in zip(axes, df_sorted.iterrows()):
        ref_path = f"{out_dir}/window_{orig_idx}_ref.npy"
        alt_path = f"{out_dir}/window_{orig_idx}_alt.npy"

        if not (os.path.exists(ref_path) and os.path.exists(alt_path)):
            ax.set_visible(False)
            continue

        ref_vals = np.load(ref_path)
        alt_vals = np.load(alt_path)

        start_0 = int(row["window_start"]) - 1
        bin_pos = start_0 + np.arange(ref_vals.shape[0]) * BIN_SIZE

        mask = (bin_pos >= x_min) & (bin_pos <= x_max)
        x = bin_pos[mask]
        ref_mean = ref_vals[mask].mean(axis=1)
        alt_mean = alt_vals[mask].mean(axis=1)

        ax.fill_between(x, ref_mean, alpha=0.45, color="steelblue", label="REF")
        ax.fill_between(x, alt_mean, alpha=0.45, color="crimson", label="ALT")
        ax.plot(x, ref_mean, color="steelblue", lw=0.7)
        ax.plot(x, alt_mean, color="crimson", lw=0.7, linestyle="--")
        ax.axvline(var_pos, color="black", lw=1, linestyle=":", alpha=0.7)

        vp = int(row["var_pos_in_window"])
        ax.set_ylabel(f"{vp} bp", fontsize=8, rotation=0, ha="right", va="center", labelpad=4)
        ax.set_yticks([])
        ax.set_xlim(x_min, x_max)
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
    axes[-1].set_xlabel("chr5 position (bp)", fontsize=10)

    fig.text(0.0, 0.5, "var_pos_in_window →", va="center", rotation=90, fontsize=9, color="dimgrey")
    fig.suptitle(
        f"TERT {variant_name} — AlphaGenome RNA-seq tracks  "
        f"(1 bp res, ±{ZOOM_BP // 1000} kb around variant)",
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

# ── PyTorch inference (CPU) ────────────────────────────────────────────────
import torch
from alphagenome_pytorch import AlphaGenome
from huggingface_hub import hf_hub_download

weights_path = hf_hub_download(
    repo_id="gtca/alphagenome_pytorch",
    filename="model_all_folds.safetensors",
    local_dir="/scratch/st-cdeboer-1/sambina/position_mpra/outputs/8-aphagenome",
)

pt_model = AlphaGenome.from_pretrained(weights_path, device="cpu")
pt_model.eval()

# One-hot encoded DNA sequence in NLC format (batch=1, length=131072, channels=4)
# Channels: A=0, C=1, G=2, T=3
sequence = np.random.randint(0, 4, size=(1, 131072))
dna_onehot = torch.tensor(np.eye(4)[sequence], dtype=torch.float32)

with torch.no_grad():
    outputs = pt_model.predict(dna_onehot, organism_index=0)  # 0=human, 1=mouse

print("PyTorch predicted outputs shape:", outputs.shape)
