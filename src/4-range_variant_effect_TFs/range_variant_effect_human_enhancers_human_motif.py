import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import FancyBboxPatch
from tqdm import tqdm
import torch
import os
import sys

sys.path.append("/scratch/st-cdeboer-1/sambina/mpra/models/random-promoter-dream-challenge-2022/benchmarks/human")
from prixfixe.autosome import AutosomeFinalLayersBlock
from prixfixe.bhi import BHIFirstLayersBlock, BHICoreBlock
from prixfixe.prixfixe import PrixFixeNet

SEQ_SIZE = 231
CUDA_DEVICE_ID = 0

def predict(df, output):
    generator = torch.Generator()
    
    #### One Hot Encoder ####
    def one_hot_encode(seq):
        mapping = {'A': [1, 0, 0, 0],
                'G': [0, 1, 0, 0],
                'C': [0, 0, 1, 0],
                'T': [0, 0, 0, 1],
                'N': [0, 0, 0, 0]}
        return [mapping.get(base.upper(), [0, 0, 0, 0]) for base in seq]

    #### Prediction #### 
    def predict_expression(model, test_df, seq_size, device, output):
        """Predict expression values using the model and save predictions line by line."""
        with open(f"{output}", 'w') as f:
     
            print(test_df.columns)
            if "motif_name" in test_df.columns:
                f.write('seq_id\tmotif_name\trandom_seq\tpredictions\n')
            else:
                f.write('seq_id,random_seq,predictions\n')
            for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
                
                #### Predict on reference sequence ####
                if len(row["random_seq"]) != 231:
                    encoded_seq_ref = one_hot_encode(row['random_seq'])
                else:
                    encoded_seq_ref = one_hot_encode(row['random_seq'])
                rev_value = [row['rev']] * len(encoded_seq_ref)
                encoded_seq_with_rev_ref = [list(encoded_base) + [rev] for encoded_base, rev in zip(encoded_seq_ref, rev_value)]

                input_tensor_ref = torch.tensor(
                    np.array(encoded_seq_with_rev_ref).reshape(1, seq_size, 5).transpose(0, 2, 1),
                    device=device,
                    dtype=torch.float32
                )

                pred_ref = model(input_tensor_ref)
                pred_value_ref = pred_ref.detach().cpu().flatten().tolist()
                
                if "motif_name" in test_df.columns:
                    f.write(f"{row['seq_id']}\t{row['motif_name']}\t{row['random_seq']}\t{pred_value_ref[0]}\n")
                else:
                    f.write(f"{row['seq_id']}\t{row['random_seq']}\t{pred_value_ref[0]}\n")


        print(f"Predictions saved to {output}")
    
    def initialize_model(seq_size, generator):
        """Initialize the PrixFixeNet model."""
        first = BHIFirstLayersBlock(
            in_channels=5,
            out_channels=320,
            seqsize=seq_size,
            kernel_sizes=[9, 15],
            pool_size=1,
            dropout=0.2
        )

        core = BHICoreBlock(
            in_channels=first.out_channels,
            out_channels=320,
            seqsize=first.infer_outseqsize(),
            lstm_hidden_channels=320,
            kernel_sizes=[9, 15],
            pool_size=1,
            dropout1=0.2,
            dropout2=0.5
        )

        final = AutosomeFinalLayersBlock(in_channels=core.out_channels)

        model = PrixFixeNet(
            first=first,
            core=core,
            final=final,
            generator=generator
        )

        return model
    
    
    def load_model_weights(model, model_path, device):
        """Load pre-trained weights into the model."""
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model = model.to(device)
        model.eval()
        return model

    device = torch.device(f"cuda:{CUDA_DEVICE_ID}" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "/scratch/st-cdeboer-1/sambina/mpra/output/chromosome/gosai/output_lfcse/output_k562/fold_4/model_best.pth"
    model = initialize_model(SEQ_SIZE, generator)    
    trained_model = load_model_weights(model, MODEL_PATH, device)
    predict_expression(trained_model, df, SEQ_SIZE, device, f"{output}")
    

### Pick top 25 percentile from here

import numpy as np
import pandas as pd

selected_df = pd.read_csv("/scratch/st-cdeboer-1/sambina/position_mpra/outputs/range/human/downsampled1000_gosai_fold4.tsv", sep="\t")

motifs_df = pd.read_csv("/scratch/st-cdeboer-1/sambina/position_mpra/outputs/TFs_position/human/k562_master_regulator.csv")
motifs = motifs_df.set_index("Seed_motif")["Consensus"].to_dict()

mutated_motifs = {key + "_alt": "N" * len(value) for key, value in motifs.items()}

# Merge original and mutated motifs into one dictionary
mutated_motifs_list = {**motifs, **mutated_motifs}
print(mutated_motifs_list)

def place_motifs_with_offsets(df, motifs, offsets=[0]):
    new_data = []

    for _, row in df.iterrows():
        sequence = row["seq"]
        seq_id = row["seq_id"]
        seq_length = len(sequence)
        center_pos = seq_length // 2  

        for motif_name, motif_seq in motifs.items():
            motif_length = len(motif_seq)

            for offset in offsets:
                start_pos = center_pos - (motif_length // 2)
                
                if start_pos < 0 or start_pos + motif_length > seq_length:
                    continue  

                new_sequence = sequence[:start_pos] + motif_seq + sequence[start_pos + motif_length:]
                new_data.append([seq_id, motif_name, new_sequence, start_pos, offset])

    new_df = pd.DataFrame(new_data, columns=["seq_id", "motif_name", "random_seq", "start_pos", "offset"])
    return new_df

dataframe = place_motifs_with_offsets(selected_df, mutated_motifs_list)
print(len(dataframe))

### Now calculate the variant effects

dataframe['rev'] = dataframe['seq_id'].str.endswith('_Reversed:') | dataframe['seq_id'].str.endswith('_R')
dataframe['rev'] = dataframe['rev'].astype(int)
predict(dataframe, "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/range/human/predicted_human_tfs_enhancer.csv")