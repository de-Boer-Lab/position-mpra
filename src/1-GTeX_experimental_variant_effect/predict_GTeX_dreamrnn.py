### Read the fasta file
from Bio import SeqIO
import gzip
import pandas as pd

fasta_path = (
    "/scratch/st-cdeboer-1/sambina/position_mpra/data/GTeX_experimental_data/ENCFF728XQT.fasta.gz"
)

records = list(SeqIO.parse(gzip.open(fasta_path, "rt"), "fasta"))
print(f"Loaded {len(records)} sequences")
print(records[0].id)

fasta_df = pd.DataFrame(
    {"id": [rec.id for rec in records], "sequence": [str(rec.seq) for rec in records]}
)

fasta_df["base_id"] = fasta_df["id"].str.extract(r"^([^:]+:[^:]+)")
print(fasta_df.head())

### Read the csv file
data_var_effects = pd.read_csv(
    "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/1-GTeX_experimental_variant_effect/42_variants.csv"
)

# Merge fasta_df with data_var_effects
merged_df = fasta_df.merge(
    data_var_effects,
    left_on="base_id",  # from fasta_df
    right_on="base_id",
)

print(merged_df.head())
print(f"Merged shape: {merged_df.shape}")

import pandas as pd
import torch
from torchinfo import summary
from tqdm import tqdm
import numpy as np
import sys
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import json
import os

sys.path.append(
    "/scratch/st-cdeboer-1/sambina/mpra/models/random-promoter-dream-challenge-2022/benchmarks/human"
)

from prixfixe.autosome import AutosomeFinalLayersBlock
from prixfixe.bhi import BHIFirstLayersBlock, BHICoreBlock
from prixfixe.prixfixe import PrixFixeNet


def configure_device(device):
    """Configures the CUDA device."""
    device = torch.device(device)
    generator = torch.Generator()
    generator.manual_seed(42)
    return device, generator


def initialize_model(seq_size, generator):
    """Initializes the PrixFixeNet model."""
    first = BHIFirstLayersBlock(
        in_channels=5,
        out_channels=320,
        seqsize=seq_size,
        kernel_sizes=[9, 15],
        pool_size=1,
        dropout=0.2,
    )

    core = BHICoreBlock(
        in_channels=first.out_channels,
        out_channels=320,
        seqsize=first.infer_outseqsize(),
        lstm_hidden_channels=320,
        kernel_sizes=[9, 15],
        pool_size=1,
        dropout1=0.2,
        dropout2=0.5,
    )

    final = AutosomeFinalLayersBlock(in_channels=core.out_channels)

    model = PrixFixeNet(first=first, core=core, final=final, generator=generator)

    return model


def load_model_weights(model, model_log_dir, device):
    """Loads model weights from the specified path."""
    print(device)
    model.load_state_dict(torch.load(model_log_dir, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    return model


def print_model_summary(model, seq_size):
    """Prints the model summary."""
    print(summary(model, (1, 5, seq_size)))


def add_reverse_column(filtered_data):
    """Add a reverse column indicating if '_Reversed:' is present in the Sequence_ID."""
    filtered_data["rev"] = filtered_data["id"].str.contains("_Reversed:").astype(int)
    return filtered_data


def one_hot_encode(seq):
    """One-hot encode a DNA sequence."""
    mapping = {
        "A": [1, 0, 0, 0],
        "G": [0, 1, 0, 0],
        "C": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "N": [0, 0, 0, 0],
    }
    return [mapping[base] for base in seq]


def encode_sequences(test_df, model_rnn, SEQ_SIZE, device):
    """One-hot encode the sequences and run predictions using the RNN model.
    If sequence length is < 200, add 'N' at the rightmost side.
    """
    encoded_seqs = []
    pred_expr_rnn = []
    for i, row in tqdm(test_df.iterrows()):
        seq = row["sequence"]
        if len(seq) < 200:
            seq = "N" * (200 - len(seq)) + seq
        seq = "ACTGGCCGCTTGACG" + seq + "CACTGCGGCTCCTGCG"

        encoded_seq = one_hot_encode(seq)
        rev_value = [row["rev"]] * len(encoded_seq)
        encoded_seq_with_rev = [
            list(encoded_base) + [rev] for encoded_base, rev in zip(encoded_seq, rev_value)
        ]

        pred = model_rnn(
            torch.tensor(
                np.array(encoded_seq_with_rev).reshape(1, SEQ_SIZE, 5).transpose(0, 2, 1),
                dtype=torch.float32,
            ).to(device)
        )
        pred_expr_rnn.append(pred.detach().cpu().flatten().tolist())

    return pred_expr_rnn


def save_predictions(test_df, pred_expr_rnn, OUTPUT):
    """Save the predicted values to a file."""
    os.makedirs(OUTPUT, exist_ok=True)

    file_path = f"{OUTPUT}/predicted_mean.txt"
    df = pd.DataFrame({"seq_id": test_df["id"], "prediction": pred_expr_rnn})

    df["prediction"] = df["prediction"].apply(lambda x: x[0])
    df.to_csv(file_path, index=False)
    return df


def main():
    CUDA_DEVICE_ID = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT = (
        "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/1-GTeX_experimental_variant_effect/"
    )
    MODEL_LOG_DIR = "/scratch/st-cdeboer-1/sambina/mpra/output/chromosome/gosai/output_lfcse/output_k562/fold_4/model_best.pth"
    sys.path.append(
        "/scratch/st-cdeboer-1/sambina/mpra/models/random-promoter-dream-challenge-2022/benchmarks/human"
    )

    SEQ_SIZE = 231

    device, generator = configure_device(CUDA_DEVICE_ID)
    model_rnn = initialize_model(SEQ_SIZE, generator)
    print_model_summary(model_rnn, SEQ_SIZE)
    model_rnn = load_model_weights(model_rnn, MODEL_LOG_DIR, device)

    filtered_data_reversed = add_reverse_column(merged_df)
    pred_expr_rnn = encode_sequences(filtered_data_reversed, model_rnn, SEQ_SIZE, device)
    predict_df = save_predictions(filtered_data_reversed, pred_expr_rnn, OUTPUT)


if __name__ == "__main__":
    main()

predictions = pd.read_csv(
    "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/1-GTeX_experimental_variant_effect/predicted_mean.txt"
)

parts = predictions["seq_id"].str.split(":", expand=True)

# Assign new columns
predictions["base_id"] = parts[0] + ":" + parts[1]
predictions["allele"] = parts[4]
predictions["window"] = parts[5]
predictions

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.stats import pearsonr
import pandas as pd
import numpy as np

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
sns.set(style="white", font_scale=3)

# Load data
pred = pd.read_csv(
    "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/1-GTeX_experimental_variant_effect/predicted_mean.txt"
)
measured = pd.read_csv(
    "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/1-GTeX_experimental_variant_effect/42_variants.csv"
)

# Parse seq_id into components
parts = pred["seq_id"].str.split(":", expand=True)
pred["base_id"] = parts[0] + ":" + parts[1]
pred["allele"] = parts[4]
pred["window"] = parts[5]

# Compute predicted variant effect (alt - ref) for each window
window_map = {"wC": "centre", "wL": "left", "wR": "right"}
rows = []
for win_code, meas_col in window_map.items():
    alt = pred[(pred["allele"] == "A") & (pred["window"] == win_code)].set_index("base_id")[
        "prediction"
    ]
    ref = pred[(pred["allele"] == "R") & (pred["window"] == win_code)].set_index("base_id")[
        "prediction"
    ]
    delta = (ref - alt).rename("pred_effect").reset_index()
    delta["window"] = win_code
    meas_sub = measured[["base_id", meas_col]].rename(columns={meas_col: "measured_effect"})
    rows.append(delta.merge(meas_sub, on="base_id"))

df = pd.concat(rows, ignore_index=True)

labels = {"wC": "Centre", "wL": "Left", "wR": "Right"}


def scatter_panel(ax, x, y, title):
    r, p = pearsonr(x, y)
    sns.scatterplot(x=x, y=y, ax=ax, color="#4C72B0", edgecolor="white", s=200, alpha=0.6)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.7)
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.set_title(f"{title}\n$r$ = {r:.3f}", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal", adjustable="datalim")
    sns.despine(ax=ax)


fig, axes = plt.subplots(2, 2, figsize=(18, 18))
axes = axes.flatten()

# Overall
scatter_panel(axes[0], df["pred_effect"], df["measured_effect"], "Overall (3 windows)")

# Per window
for ax, (win_code, meas_col) in zip(axes[1:], window_map.items()):
    grp = df[df["window"] == win_code]
    scatter_panel(ax, grp["pred_effect"], grp["measured_effect"], f"{labels[win_code]} window")

plt.tight_layout(pad=4.0)

out_path = "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/1-GTeX_experimental_variant_effect/measured_vs_predicted.svg"
fig.savefig(out_path, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.show()
