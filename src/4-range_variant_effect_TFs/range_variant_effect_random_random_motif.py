import random
import pandas as pd
import numpy as np
import os
import torch
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# ============ Load PrixFixe Imports ============
sys.path.append("/scratch/st-cdeboer-1/sambina/mpra/models/random-promoter-dream-challenge-2022/benchmarks/human")
from prixfixe.autosome import AutosomeFinalLayersBlock
from prixfixe.bhi import BHIFirstLayersBlock, BHICoreBlock
from prixfixe.prixfixe import PrixFixeNet

# ============ Paths ============
SEQ_SIZE = 231
CUDA_DEVICE_ID = 0
MODEL_PATH = "/scratch/st-cdeboer-1/sambina/mpra/output/chromosome/gosai/output_lfcse/output_k562/fold_4/model_best.pth"
OUTPUT_DIR = "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/4-range/human"
os.makedirs(OUTPUT_DIR, exist_ok=True)

input_prediction_file = f"{OUTPUT_DIR}/random_sequences.csv"
motif_output = f"{OUTPUT_DIR}/predicted_random_motif_control.csv"

# ============ Helper Functions ============

def one_hot_encode(seq):
    mapping = {'A': [1, 0, 0, 0],
               'G': [0, 1, 0, 0],
               'C': [0, 0, 1, 0],
               'T': [0, 0, 0, 1],
               'N': [0, 0, 0, 0]}
    return [mapping.get(base.upper(), [0, 0, 0, 0]) for base in seq]

def initialize_model(seq_size, generator):
    first = BHIFirstLayersBlock(
        in_channels=5, out_channels=320, seqsize=seq_size,
        kernel_sizes=[9, 15], pool_size=1, dropout=0.2
    )
    core = BHICoreBlock(
        in_channels=first.out_channels, out_channels=320,
        seqsize=first.infer_outseqsize(), lstm_hidden_channels=320,
        kernel_sizes=[9, 15], pool_size=1, dropout1=0.2, dropout2=0.5
    )
    final = AutosomeFinalLayersBlock(in_channels=core.out_channels)
    return PrixFixeNet(first=first, core=core, final=final, generator=generator)

def load_model_weights(model, model_path, device):
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model = model.to(device)
    model.eval()
    return model

def predict_expression(model, test_df, seq_size, device, output):
    with open(output, 'w') as f:
        upstream = "ACTGGCCGCTTGACG"
        downstream = "CACTGCGGCTCCTGCG"
        if "motif_name" in test_df.columns:
            f.write('seq_id,motif_name,random_seq,predictions\n')
        else:
            f.write('seq_id,random_seq,predictions\n')

        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            seq = row["random_seq"]
            if len(seq) != 231:
                seq = upstream + seq + downstream

            encoded_seq = one_hot_encode(seq)
            rev_value = [row.get('rev', 0)] * len(encoded_seq)
            encoded_seq_with_rev = [list(base) + [rev] for base, rev in zip(encoded_seq, rev_value)]
            input_tensor = torch.tensor(
                np.array(encoded_seq_with_rev).reshape(1, seq_size, 5).transpose(0, 2, 1),
                device=device, dtype=torch.float32
            )

            pred = model(input_tensor)
            pred_value = pred.detach().cpu().flatten().tolist()[0]
            if "motif_name" in test_df.columns:
                f.write(f"{row['seq_id']},{row['motif_name']},{seq},{pred_value}\n")
            else:
                f.write(f"{row['seq_id']},{seq},{pred_value}\n")


def predict(df, output):
    generator = torch.Generator()
    device = torch.device(f"cuda:{CUDA_DEVICE_ID}" if torch.cuda.is_available() else "cpu")
    model = initialize_model(SEQ_SIZE, generator)
    trained_model = load_model_weights(model, MODEL_PATH, device)
    predict_expression(trained_model, df, SEQ_SIZE, device, output)
    print(f"Predictions saved to {output}")


# ============ Step 1: Read existing random predictions ============
prediction = pd.read_csv(input_prediction_file)
print(f"Loaded {len(prediction)} random sequence predictions.")

# Optional: visualize
plt.figure(figsize=(6, 4))
sns.boxplot(y=prediction['predictions'])
plt.ylabel("Predicted Expression")
plt.title("Random Sequence Predictions")
plt.show()

# ============ Step 2: Select top 25% high-expression sequences ============
def select_percentile_sequences(df, num_sequences=500):
    upper_bound = df['predictions'].quantile(0.75)
    filtered_df = df[df['predictions'] >= upper_bound]
    selected = filtered_df.sample(n=min(num_sequences, len(filtered_df)), random_state=42)
    return selected[['seq_id', 'random_seq', 'predictions']]

selected_df = select_percentile_sequences(prediction)
print(f"Selected {len(selected_df)} high-expression sequences.")

# ============ Step 3: Create random motif and insert ============
avg_motif_length = 14
def generate_random_motif(length, seed=42):
    random.seed(seed)
    return ''.join(random.choices(['A', 'T', 'C', 'G'], k=length))

random_motif = generate_random_motif(avg_motif_length)
print(random_motif)
motifs = {
    "random_motif": random_motif,
    "random_motif_alt": "N" * avg_motif_length
}
print("Using motifs:", motifs)

def place_motifs(df, motifs):
    new_data = []
    for _, row in df.iterrows():
        seq = row["random_seq"]
        seq_id = row["seq_id"]
        seq_length = len(seq)
        center = seq_length // 2

        for motif_name, motif_seq in motifs.items():
            start = center - len(motif_seq) // 2
            new_seq = seq[:start] + motif_seq + seq[start + len(motif_seq):]
            new_data.append([seq_id, motif_name, new_seq, start, 0])
    return pd.DataFrame(new_data, columns=["seq_id", "motif_name", "random_seq", "start_pos", "offset"])

motif_df = place_motifs(selected_df, motifs)
motif_df['rev'] = 0
print(f"Placed motifs in {len(motif_df)} total sequences (ref + alt).")

# ============ Step 4: Predict for motif-inserted sequences ============
predict(motif_df, motif_output)
print("Random motif predictions complete!")
