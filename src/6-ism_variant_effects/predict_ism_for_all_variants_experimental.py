import pandas as pd
from Bio import SeqIO
import gzip

### First we get the wL, wC and wC of the 43 variants (both Ref and Alt)

df = pd.read_csv("/scratch/st-cdeboer-1/sambina/position_mpra/outputs/1-GTeX_experimental_variant_effect/42_variants.csv")
fasta_path = "/scratch/st-cdeboer-1/sambina/position_mpra/data/GTeX_experimental_data/ENCFF728XQT.fasta.gz"

records = []
with gzip.open(fasta_path, "rt") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        records.append({
            "id": record.id,
            "description": record.description,
            "sequence": str(record.seq)
        })

fasta_df = pd.DataFrame(records)

print(df.head())
print(fasta_df.head())

def check_seq_id(s):
    parts = s.split(":")
    # make sure we have enough parts
    if len(parts) >= 4:
        return len(parts[2]) == 1 and len(parts[3]) == 1
    return False

fasta_df = fasta_df[fasta_df["id"].apply(check_seq_id)]
fasta_df.shape

### create base_id from fasta_df 

fasta_df["base_id"] = fasta_df["id"].apply(lambda x: ":".join(x.split(":")[:2]))
fasta_df.head(2)

### merge fasta_df with the 43 variants

merged = pd.merge(df, fasta_df, on="base_id", how="inner")
print(merged.head())
print(merged.shape)

### Function that takes a seq and outputs a prediction value
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import json
import sys
import os
import matplotlib.pyplot as plt
import logomaker

sys.path.append("/scratch/st-cdeboer-1/sambina/mpra/models/random-promoter-dream-challenge-2022/benchmarks/human")
TRAIN_BATCH_SIZE = 32
N_PROCS = 4
VALID_BATCH_SIZE = 32
SEQ_SIZE = 231
generator = torch.Generator()

from prixfixe.autosome import AutosomeFinalLayersBlock
from prixfixe.bhi import BHIFirstLayersBlock
from prixfixe.bhi import BHICoreBlock
from prixfixe.prixfixe import PrixFixeNet

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

upstream = "ACTGGCCGCTTGACG"
downstream = "CACTGCGGCTCCTGCG"

def one_hot_encode(seq):
    mapping = {'A': [1, 0, 0, 0],
               'G': [0, 1, 0, 0],
               'C': [0, 0, 1, 0],
               'T': [0, 0, 0, 1],
               'N': [0, 0, 0, 0]}
    return [mapping.get(base.upper(), [0, 0, 0, 0]) for base in seq]

def predict_fn(seq):
    rev_flag = 0
    SEQ_SIZE = 231  
    CUDA_DEVICE_ID = 0
    device = torch.device(f"cuda:{CUDA_DEVICE_ID}" if torch.cuda.is_available() else "cpu")
    

    #### loading the model #### 
    first = BHIFirstLayersBlock(in_channels=5, out_channels=320, seqsize=SEQ_SIZE, kernel_sizes=[9, 15], pool_size=1, dropout=0.2)
    core = BHICoreBlock(in_channels=first.out_channels, out_channels=320, seqsize=first.infer_outseqsize(), lstm_hidden_channels=320, kernel_sizes=[9, 15], pool_size=1, dropout1=0.2, dropout2=0.5)
    final = AutosomeFinalLayersBlock(in_channels=core.out_channels)

    model_rnn = PrixFixeNet(first=first, core=core, final=final, generator=generator)
    model_path = "/scratch/st-cdeboer-1/sambina/mpra/output/chromosome/gosai/output_lfcse/output_k562/fold_0/model_best.pth"
    model_rnn.load_state_dict(torch.load(model_path, map_location=device))
    model_rnn.to(device)
    model_rnn.eval()
    
    seq_length = len(seq)

    encoded_seq = one_hot_encode(seq)
    rev_values = [rev_flag] * seq_length
    encoded_seq_with_rev = [list(base) + [rev] for base, rev in zip(encoded_seq, rev_values)]
    original_input = torch.tensor(np.array(encoded_seq_with_rev).reshape(1, seq_length, 5).transpose(0, 2, 1),
                                  device=device, dtype=torch.float32)
    
    with torch.no_grad():
        original_prediction = model_rnn(original_input).cpu().flatten().item()
        
    return original_prediction  

### predict for 4 mutations per sequence per position = number of preds = 4 * L of seq * # of seq

BASES = ["A", "C", "G", "T"]

all_results = []

for idx, row in tqdm(merged.iterrows(), total=len(merged), desc="Predicting mutations"):
    seq = row["sequence"]
    seq_id = row["id"]

    base_pred = predict_fn(seq)

    mutation_results = []

    for pos in range(len(seq)):
        ref_base = seq[pos]

        position_dict = {f"{ref_base}_ref_base": base_pred}

        for b in BASES:
            if b != ref_base:
                mutated_seq = seq[:pos] + b + seq[pos + 1:]
                mut_pred = predict_fn(mutated_seq)
                position_dict[b] = mut_pred

        mutation_results.append(position_dict)

    all_results.append({
        "seq_id": seq_id,
        "mutations": mutation_results
    })
    

mut_df = pd.DataFrame(all_results)

### calculate ISM wrt to the original prediction for each base 

def compute_diffs(mutation_list):
    """Takes a list of dicts (each position), returns list of dicts of deltas."""
    diffs = []
    for d in mutation_list:
        ref_key = [k for k in d.keys() if k.endswith("_ref_base")][0]
        ref_base = ref_key.split("_")[0]
        ref_val = d[ref_key]
        
        diff_dict = {}
        for base, val in d.items():
            if base == ref_key:
                continue
            diff_dict[base] = ref_val - val

        diff_dict["ref_base"] = ref_base
        diffs.append(diff_dict)
    return diffs


# Apply to every row of mut_df
mut_df["ism_predictions"] = mut_df["mutations"].apply(compute_diffs)

print(mut_df["ism_predictions"])

### Adding a new column for ref or alt

mut_df["base_id"] = mut_df["seq_id"].apply(
    lambda s: ":".join(s.split(":")[:-2] + [s.split(":")[-1]])
)

mut_df["allele_type"] = mut_df["seq_id"].apply(
    lambda s: "ref" if s.split(":")[-2] == "R" else ("alt" if s.split(":")[-2] == "A" else "unknown")
)

### Calculate change in ISM for each position now. We will end up with 3 values. if its the variant position we only do variant effect (might need to revise this)

def compute_delta(ref_preds, alt_preds):
    delta = []
    for r_dict, a_dict in zip(ref_preds, alt_preds):
        r_base = r_dict["ref_base"]
        a_base = a_dict["ref_base"]
        new_entry = {}

        if r_base == a_base:
            # Same reference base — simple subtraction
            for base in ['A', 'C', 'G', 'T']:
                if base in r_dict and base in a_dict:
                    new_entry[base] = r_dict[base] - a_dict[base]
            new_entry["ref_base"] = r_base
        else:
            # Different ref bases — subtract matching ones and handle ref_base separately
            for base in ['A', 'C', 'G', 'T']:
                if base in r_dict and base in a_dict:
                    new_entry[base] = r_dict[base] - a_dict[base]
                else:
                    new_entry[f"{r_base}-{a_base}"] = r_dict[a_base] # because I calculated the var effect 
            new_entry["ref_base"] = f"{r_base}-{a_base}"  # just to track what happened
        delta.append(new_entry)
    return delta

def compute_group_delta(group):
    ref_row = group[group["allele_type"] == "ref"].iloc[0]
    alt_row = group[group["allele_type"] == "alt"].iloc[0]
    return compute_delta(ref_row["ism_predictions"], alt_row["ism_predictions"])

### Apply groupby + delta computation
delta_df = (
    mut_df.groupby("base_id")
    .apply(lambda g: compute_group_delta(g) if len(g) == 2 else None)
    .reset_index(name="delta_predictions")
)

### Now lets calculate avg ism per position so each sequence has a vector of L values
def compute_position_means(delta_list):
    if not isinstance(delta_list, list):
        return None  

    position_means = []
    for d in delta_list:
        float_vals = [v for v in d.values() if isinstance(v, (int, float))]
        if float_vals:
            position_means.append(sum(float_vals) / len(float_vals))
        else:
            position_means.append(None)
    return position_means

delta_df["ism"] = delta_df["delta_predictions"].apply(compute_position_means)
delta_df["ism"].apply(lambda x: len(x)).unique()

# Find rows where ism length != 200
mask = delta_df["ism"].apply(lambda x: len(x) if isinstance(x, list) else 0) != 200

# Show the rows
delta_df[mask]
delta_df.loc[mask, "base_id"]

delta_df.to_csv("/scratch/st-cdeboer-1/sambina/position_mpra/outputs/6-ism_variant_effects/delta_ism.csv")

# normalize ism
# Convert column to list of arrays (if not already)
ism_arrays = delta_df["ism"].tolist()

# Find the global max absolute value across all arrays
max_abs_val = max(np.max(np.abs(arr)) for arr in ism_arrays)
print(f"Max absolute ISM value: {max_abs_val:.4f}")

# Divide each ISM array by that value
# delta_df["ism"] = [arr / max_abs_val for arr in ism_arrays]
delta_df["ism"] = [arr / 1 for arr in ism_arrays]


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- Make sure base_id exists ---
assert "base_id" in delta_df.columns, "base_id column missing!"

# --- Split out parts of base_id ---
base_split = delta_df["base_id"].str.split(":")
delta_df["id_prefix"] = base_split.str[:2].str.join(":")   
delta_df["id_suffix"] = base_split.str[-1]                 

# --- Filter only expected suffixes ---
valid_suffixes = ["wC", "wR", "wL"]
delta_df = delta_df[delta_df["id_suffix"].isin(valid_suffixes)]

# --- Check which suffixes exist ---
present_suffixes = sorted(delta_df["id_suffix"].unique())
print(f"✅ Found suffix groups: {present_suffixes}")

# --- Print counts per suffix ---
for suffix in valid_suffixes:
    count = (delta_df["id_suffix"] == suffix).sum()
    print(f"Rows with {suffix}: {count}")

# --- Ensure consistent ordering across suffixes ---
# keep only rows whose prefix appears in *all* three
prefix_sets = {
    suffix: set(delta_df.loc[delta_df["id_suffix"] == suffix, "id_prefix"])
    for suffix in present_suffixes
}
common_ids = set.intersection(*prefix_sets.values()) if len(prefix_sets) > 1 else set()

print(f"\nCommon id_prefix entries across all groups: {len(common_ids)}")

if not common_ids:
    raise ValueError("No common id_prefix entries across all wC/wR/wL groups.")

# --- Filter for common ids only ---
delta_df = delta_df[delta_df["id_prefix"].isin(common_ids)]

# --- Sort by id_prefix for consistent row order ---
delta_df = delta_df.sort_values(["id_prefix", "id_suffix"])

# --- Build dict of suffix→data ---
dfs = {
    suffix: delta_df[delta_df["id_suffix"] == suffix]
    for suffix in present_suffixes
}

# --- Print lengths of each df explicitly ---
for k, v in dfs.items():
    print(f"{k} DataFrame length: {len(v)}")

# --- Get row labels (same for all suffixes) ---
row_labels = dfs[present_suffixes[0]]["id_prefix"].tolist()

# --- Convert ISM column to consistent arrays ---
heatmaps = {}
for k, v in dfs.items():
    arr_list = v["ism"].tolist()
    max_len = max(len(a) for a in arr_list)
    print(f"{k}: ISM array length (per row): {max_len}")
    arr = np.array([np.pad(a, (0, max_len - len(a))) for a in arr_list])
    heatmaps[k] = arr


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- Reorder so wC is in the middle ---
order = [s for s in ["wL", "wC", "wR"] if s in present_suffixes]
n = len(order)

# --- Shared color scale across all heatmaps ---
vmin = min(np.min(heatmaps[s]) for s in order)
vmax = max(np.max(heatmaps[s]) for s in order)

# --- Create subplots ---
fig, axes = plt.subplots(1, n, figsize=(18, 10), sharey=True)

# If only one suffix, axes won't be iterable
if n == 1:
    axes = [axes]

# --- Plot each heatmap ---
for i, suffix in enumerate(order):
    ax = axes[i]
    hm = sns.heatmap(
        heatmaps[suffix],
        cmap="coolwarm",
        center=0,
        vmin=vmin,
        vmax=vmax,
        xticklabels=10,
        yticklabels=row_labels, 
        cbar=(i == n - 1),  # only add colorbar on last plot
        ax=ax
    )

    # Axis/labels
    ax.set_title(f"ISM Δ ({suffix})")
    ax.set_xlabel("Position")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    if i == 0:
        ax.set_ylabel("Variant")
    else:
        ax.set_ylabel("")

# --- Adjust layout ---
plt.tight_layout()
plt.savefig("/scratch/st-cdeboer-1/sambina/position_mpra/outputs/6-ism_variant_effects/ism_all.png")
plt.show()

