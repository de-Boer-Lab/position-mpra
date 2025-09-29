import sys
sys.path.append("/scratch/st-cdeboer-1/sambina/mpra/mpra_models/random-promoter-dream-challenge-2022/benchmarks/human")
print(sys.path)

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prixfixe.autosome import AutosomeFinalLayersBlock, AutosomeDataProcessor, AutosomeTrainer
from prixfixe.bhi import BHIFirstLayersBlock, BHICoreBlock
from prixfixe.prixfixe import PrixFixeNet
from torchinfo import summary
from scipy.stats import pearsonr, spearmanr
import json

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

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
    model.eval()
    return model


def preprocess_data(data_path):
    """Read and preprocess the data."""
    df = pd.read_csv(data_path, sep="\t")
    df['rev'] = df['seq_id'].str.endswith('_Reversed:') | df['seq_id'].str.endswith('_R')
    df['rev'] = df['rev'].astype(int)
    return df


def one_hot_encode(seq):
    """One-hot encode a DNA sequence."""
    mapping = {
        'A': [1, 0, 0, 0],
        'G': [0, 1, 0, 0],
        'C': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]
    }
    return [mapping[base] for base in seq]

def predict_expression(model, seq, seq_size, device):
    """Predict expression values for a given sequence."""
    encoded_seq = one_hot_encode(seq)
    rev_value = [0 if 'rev' not in seq else 1] * len(encoded_seq)
    encoded_seq_with_rev = [list(encoded_base) + [rev] for encoded_base, rev in zip(encoded_seq, rev_value)]

    input_tensor = torch.tensor(
        np.array(encoded_seq_with_rev).reshape(1, seq_size, 5).transpose(0, 2, 1),
        device=device,
        dtype=torch.float32
    )
    
    pred = model(input_tensor)
    pred_value = pred.detach().cpu().flatten().tolist()

    return pred_value[0]

offsets=range(-80,81)


from tqdm import tqdm  
tqdm.pandas()

def train_predict():
    """Main function to train, predict, and evaluate."""
    CUDA_DEVICE_ID = 0
    SEQ_SIZE = 231
    MODEL_PATH = f"/scratch/st-cdeboer-1/sambina/mpra/mpra_with_chromosome/gosai_2024/output_lfcse/output_k562/fold_4/model_best.pth"
    device = torch.device(f"cuda:{CUDA_DEVICE_ID}")
    generator = torch.Generator().manual_seed(42)
    model = initialize_model(SEQ_SIZE, generator)
    print(summary(model, (1, 5, SEQ_SIZE)))
    test_df = pd.read_csv("/scratch/st-cdeboer-1/sambina/position_mpra/outputs/TFs_position/human/enhancer/k562_regulator_knockout_enhancer_rc.txt", sep="\t")
    
    for offset in offsets:
        column_name = f"seq_{offset}"
        predicted_column_name = f"predicted_{offset}"

        trained_model = load_model_weights(model, MODEL_PATH, device)

        test_df[predicted_column_name] = test_df[column_name].progress_apply(
            lambda seq: predict_expression(trained_model, seq, SEQ_SIZE, device)
        )
        
    test_df.to_csv("/scratch/st-cdeboer-1/sambina/position_mpra/outputs/TFs_position/human/enhancer/predicted_k562_regulator_knockout_gosai_enhancer_rc.txt", sep="\t", index=False)
    print(test_df.head())



def main():
    train_predict()

if __name__ == "__main__":
    main()
