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

#### Sequence I am doing ISM on ####
sequence_dict = {
"A_wC": upstream + "TCTCCTCCAGGATTACTACTGTTAGTCTGTCTTTCCACCTCCAGTCTCTTGTGCCAATCCATCCCAAACATAATAGTTACAGATTGGCCGGGCGCGGTGCCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCAAGGCGGATGGATCAGCTGAAGTCAGGATCGAACGACCAGCCTGGCCAACATGGTGAAACCTTGTCT" + downstream, 
"R_wC": upstream + "TCTCCTCCAGGATTACTACTGTTAGTCTGTCTTTCCACCTCCAGTCTCTTGTGCCAATCCATCCCAAACATAATAGTTACAGATTGGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCAAGGCGGATGGATCAGCTGAAGTCAGGATCGAACGACCAGCCTGGCCAACATGGTGAAACCTTGTCT" + downstream,
"A_wL": upstream + "GTGCCAATCCATCCCAAACATAATAGTTACAGATTGGCCGGGCGCGGTGCCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCAAGGCGGATGGATCAGCTGAAGTCAGGATCGAACGACCAGCCTGGCCAACATGGTGAAACCTTGTCTCTACTAAAAATACAAATATTAGCCAGGCGTCGTCGTGGGTGCCTGTGATC" + downstream, 
"R_wL": upstream + "GTGCCAATCCATCCCAAACATAATAGTTACAGATTGGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCAAGGCGGATGGATCAGCTGAAGTCAGGATCGAACGACCAGCCTGGCCAACATGGTGAAACCTTGTCTCTACTAAAAATACAAATATTAGCCAGGCGTCGTCGTGGGTGCCTGTGATC" + downstream, 
"A_wR": upstream + "TTACATTTAAAATATTCCACATTCAGAGTTGTAGAGGCCTTAGACTATTATCTCCTCCAGGATTACTACTGTTAGTCTGTCTTTCCACCTCCAGTCTCTTGTGCCAATCCATCCCAAACATAATAGTTACAGATTGGCCGGGCGCGGTGCCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCAAGGCGGATGGATCAGC" + downstream,
"R_wR": upstream + "TTACATTTAAAATATTCCACATTCAGAGTTGTAGAGGCCTTAGACTATTATCTCCTCCAGGATTACTACTGTTAGTCTGTCTTTCCACCTCCAGTCTCTTGTGCCAATCCATCCCAAACATAATAGTTACAGATTGGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCAAGGCGGATGGATCAGC" + downstream}

#### One Hot Encoder ####
def one_hot_encode(seq):
    mapping = {'A': [1, 0, 0, 0],
               'G': [0, 1, 0, 0],
               'C': [0, 0, 1, 0],
               'T': [0, 0, 0, 1],
               'N': [0, 0, 0, 0]}
    return [mapping.get(base.upper(), [0, 0, 0, 0]) for base in seq]

#### Perform ISM per sequence, returns both importance score and mutated predictions ####
def perform_ism(sequence, model, rev_flag, device):
    sequence_length = len(sequence)

    encoded_seq = one_hot_encode(sequence)
    rev_values = [rev_flag] * sequence_length
    encoded_seq_with_rev = [list(base) + [rev] for base, rev in zip(encoded_seq, rev_values)]
    original_input = torch.tensor(np.array(encoded_seq_with_rev).reshape(1, sequence_length, 5).transpose(0, 2, 1),
                                  device=device, dtype=torch.float32)
    
    with torch.no_grad():
        original_prediction = model(original_input).cpu().flatten().item()

    importance_scores = np.zeros(sequence_length)

    for i in range(sequence_length):
        original_base = sequence[i]
        for mutated_base in "ACGT":
            if mutated_base == original_base:
                continue
            mutated_sequence = sequence[:i] + mutated_base + sequence[i+1:]
            encoded_mut = one_hot_encode(mutated_sequence)
            encoded_mut_with_rev = [list(base) + [rev_flag] for base in encoded_mut]
            mut_input = torch.tensor(np.array(encoded_mut_with_rev).reshape(1, sequence_length, 5).transpose(0, 2, 1),
                                     device=device, dtype=torch.float32)
            with torch.no_grad():
                mut_pred = model(mut_input).cpu().flatten().item()
            importance_scores[i] += (original_prediction - mut_pred)

    return importance_scores / 3.0, original_prediction

#### plot logo in one panel ####
def plot_combined_ism_logos(sequences, scores_dict, save_path=None):
    color_scheme = {
        'A': '#1f77b4',  
        'C': '#2ca02c', 
        'G': '#ff7f0e',  
        'T': '#d62728',  
    }

    seq_order = ['R_wL', 'A_wL', 'R_wC', 'A_wC', 'R_wR', 'A_wR']
    fig, axes = plt.subplots(3, 2, figsize=(12, 6), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, seq_id in enumerate(seq_order):
        sequence = sequences[seq_id]
        scores = scores_dict[seq_id]

        trimmed_seq = sequence[15:-16]
        trimmed_scores = scores[15:-16]

        df = pd.DataFrame(0.0, index=range(len(trimmed_seq)), columns=["A", "C", "G", "T"])
        for i, base in enumerate(trimmed_seq):
            df.at[i, base] = trimmed_scores[i]

        ax = axes[idx]
        logo = logomaker.Logo(df, ax=ax, color_scheme=color_scheme, center_values=False)

        logo.style_spines(visible=False)
        logo.style_spines(spines=['left', 'bottom'], visible=True)
        if seq_id.endswith("wL") & seq_id.startswith("R"):
            ax.set_title("ref variant: upstream", fontsize=12)
        elif seq_id.endswith("wC") & seq_id.startswith("R"):
            ax.set_title("ref variant: centre", fontsize=12)
        elif seq_id.endswith("wR") & seq_id.startswith("R"):
            ax.set_title("ref variant: downstream", fontsize=12)
            
        if seq_id.endswith("wL") & seq_id.startswith("A"):
            ax.set_title("alt variant: upstream", fontsize=12)
        elif seq_id.endswith("wC") & seq_id.startswith("A"):
            ax.set_title("alt variant: centre", fontsize=12)
        elif seq_id.endswith("wR") & seq_id.startswith("A"):
            ax.set_title("alt variant: downstream", fontsize=12)
            
        ax.set_xlabel("Genomic Position (hg38)", fontsize=10)
        ax.set_ylabel("Importance Score", fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_ylim(-2, 3)
        
        genomic_offset = 44139190 - 99
        xticks = ax.get_xticks()
        xtick_labels = [str(int(x + genomic_offset)) for x in xticks]
        ax.set_xticklabels(xtick_labels)

        # if seq_id.endswith("wL"):
        #     ax.axvline(x=50-1, color='red', linestyle='dotted', linewidth=1.2)
        # elif seq_id.endswith("wC"):
        #     ax.axvline(x=100-1, color='red', linestyle='dotted', linewidth=1.2)
        # elif seq_id.endswith("wR"):
        #     ax.axvline(x=150-1, color='red', linestyle='dotted', linewidth=1.2)
        highlight_positions = {
            "wL": 50 - 1,
            "wC": 100 - 1,
            "wR": 150 - 1,
        }
        
        #### This is for aesthetics only, marking the position of the variant #### 
        for key, x in highlight_positions.items():
            if seq_id.endswith(key):
                if seq_id.startswith("A_"):
                    ax.axvspan(x - 0.5, x + 0.5, color='lightcoral', alpha=0.4)
                
                if seq_id.startswith("R_"):
                    ax.axvspan(x - 0.5, x + 0.5, color='lightgreen', alpha=0.4)
                

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, format="svg")
        plt.show()


#### Now running the ISM and logo plot ####
SEQ_SIZE = 231  
CUDA_DEVICE_ID = 0
device = torch.device(f"cuda:{CUDA_DEVICE_ID}" if torch.cuda.is_available() else "cpu")

#### loading the model #### 
first = BHIFirstLayersBlock(in_channels=5, out_channels=320, seqsize=SEQ_SIZE, kernel_sizes=[9, 15], pool_size=1, dropout=0.2)
core = BHICoreBlock(in_channels=first.out_channels, out_channels=320, seqsize=first.infer_outseqsize(), lstm_hidden_channels=320, kernel_sizes=[9, 15], pool_size=1, dropout1=0.2, dropout2=0.5)
final = AutosomeFinalLayersBlock(in_channels=core.out_channels)

model_rnn = PrixFixeNet(first=first, core=core, final=final, generator=generator)
model_path = "/scratch/st-cdeboer-1/sambina/mpra/output/chromosome/gosai/output_lfcse/output_k562/fold_4/model_best.pth"
model_rnn.load_state_dict(torch.load(model_path, map_location=device))
model_rnn.to(device)
model_rnn.eval()

output_dir = "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/ISM/revcomp"
output_logo_dir = f"{output_dir}/ism_logos"
os.makedirs(output_logo_dir, exist_ok=True)

sequences = {}
scores_dict = {}

predictions = {}
for seq_id, seq in sequence_dict.items():
    print(f"Processing {seq_id}")
    rev_flag = 0 if seq_id.startswith("R_") else 0
    ism_scores, original_pred = perform_ism(seq, model_rnn, rev_flag, device)
    # 17:44139190:G:C
    offset = 44139190 - 99  # → 44139091

    df = pd.DataFrame({
        "position": np.arange(len(seq)) + offset,
        "ISM_score": ism_scores
    })
    
    df.to_csv(os.path.join(output_logo_dir, f"{seq_id}_ISM.csv"), index=False)
    print(f"Saved ISM scores to {seq_id}_ISM.csv")
    predictions[seq_id] = original_pred

    sequences[seq_id] = seq
    scores_dict[seq_id] = ism_scores
    
df_pred = pd.DataFrame(list(predictions.items()), columns=["seq_id", "original_pred"])
df_pred.to_csv(f"{output_logo_dir}/bar.csv", index=False)
plot_combined_ism_logos(sequences, scores_dict, save_path=f"{output_logo_dir}/ISM_logo_panel.svg")