### Select the variants with strong variant effect

### Get the 103k sequences and predict expression for when the variant is at the centre

import pandas as pd
import ast

# --- Load predictions file ---
predictions = pd.read_csv(
    "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/2-opentargets_model_variant_effect/gosai/k562_gosai_ref_alt_with_seq.csv.gz",
    compression="gzip"
)

# --- Convert stringified dicts in 'offset_0' column to real dicts ---
predictions["offset_0"] = predictions["offset_0"].apply(ast.literal_eval)

# --- Extract ref and alt numeric values from offset_0 dict ---
predictions["ref"] = predictions["offset_0"].apply(lambda x: x["ref"])
predictions["alt"] = predictions["offset_0"].apply(lambda x: x["alt"])

# --- Compute variant effect ---
predictions["variant_effect"] = predictions["ref"] - predictions["alt"]

# --- Extract ref/alt sequences ---
predictions["ref_seq"] = predictions["offset_0"].apply(lambda x: x["ref_seq"])
predictions["alt_seq"] = predictions["offset_0"].apply(lambda x: x["alt_seq"])

# --- Filter strong variant effects ---
filtered_variants = predictions[
    (predictions["variant_effect"] > 0.5) | (predictions["variant_effect"] < -0.5)
]

print(filtered_variants.shape)

# filtered_variants = filtered_variants.head(100)

### Predict for 4 mutations per sequence for ref and alt

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

BASES = ["A", "C", "G", "T"]

all_results = []

for idx, row in tqdm(filtered_variants.iterrows(), total=len(filtered_variants), desc="Predicting mutations"):
    
    seq = row["ref_seq"]
    base_pred = predict_fn(seq)
    mutation_results_ref = []

    for pos in range(len(seq)):
        ref_base = seq[pos]
        position_dict = {f"{ref_base}_ref_base": base_pred}

        for b in BASES:
            if b != ref_base:
                mutated_seq = seq[:pos] + b + seq[pos + 1:]
                mut_pred = predict_fn(mutated_seq)
                position_dict[b] = mut_pred

        mutation_results_ref.append(position_dict)

    seq = row["alt_seq"]
    base_pred = predict_fn(seq)
    mutation_results_alt = []

    for pos in range(len(seq)):
        ref_base = seq[pos]
        position_dict = {f"{ref_base}_alt_base": base_pred}

        for b in BASES:
            if b != ref_base:
                mutated_seq = seq[:pos] + b + seq[pos + 1:]
                mut_pred = predict_fn(mutated_seq)
                position_dict[b] = mut_pred

        mutation_results_alt.append(position_dict)
    
    all_results.append({
        "mutations_ref": mutation_results_ref,
        "mutations_alt": mutation_results_alt
    })
    

mut_df = pd.DataFrame(all_results)

### Calculate ISM wrt to the original prediction for each base 

def compute_diffs(mutation_list):
    """Takes a list of dicts (each position), returns list of dicts of deltas."""
    diffs = []
    for d in mutation_list:
        ref_key = [k for k in d.keys() if k.endswith(("_ref_base", "_alt_base"))][0]
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
mut_df["ism_predictions_ref"] = mut_df["mutations_ref"].apply(compute_diffs)
mut_df["ism_predictions_alt"] = mut_df["mutations_alt"].apply(compute_diffs)


### Calculate change in ISM for each position now. We will end up with 3 values.
import pandas as pd

def compute_delta(ref_preds, alt_preds):
    delta = []
    for r_dict, a_dict in zip(ref_preds, alt_preds):
        r_base = r_dict.get("ref_base")
        a_base = a_dict.get("ref_base")
        new_entry = {}

        if r_base == a_base:
            # same reference base — simple subtraction
            for base in ["A", "C", "G", "T"]:
                if base in r_dict and base in a_dict:
                    new_entry[base] = r_dict[base] - a_dict[base]
            new_entry["ref_base"] = r_base
        else:
            # different ref bases
            for base in ["A", "C", "G", "T"]:
                if base in r_dict and base in a_dict:
                    new_entry[base] = r_dict[base] - a_dict[base]
            new_entry[f"{r_base}-{a_base}"] = r_dict.get(a_base, None)
            new_entry["ref_base"] = f"{r_base}-{a_base}"
        delta.append(new_entry)
    return delta


# --- apply per row ---
mut_df["delta_predictions"] = mut_df.apply(
    lambda row: compute_delta(row["ism_predictions_ref"], row["ism_predictions_alt"]),
    axis=1
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

mut_df["ism"] = mut_df["delta_predictions"].apply(compute_position_means)
mut_df["ism"].apply(lambda x: len(x)).unique()

mut_df.to_csv("/scratch/st-cdeboer-1/sambina/position_mpra/outputs/6-ism_variant_effects/delta_ism_OT_all.csv")