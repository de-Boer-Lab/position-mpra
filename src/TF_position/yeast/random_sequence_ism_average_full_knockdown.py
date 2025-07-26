import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from tqdm import tqdm
import torch
import os
import sys

sys.path.append("/scratch/st-cdeboer-1/sambina/mpra/mpra_models/random-promoter-dream-challenge-2022")
from prixfixe.autosome import AutosomeDataProcessor, AutosomeFirstLayersBlock, AutosomeCoreBlock, AutosomeFinalLayersBlock, AutosomeTrainer, AutosomePredictor
from prixfixe.bhi import BHIFirstLayersBlock, BHICoreBlock
from prixfixe.unlockdna import UnlockDNACoreBlock
from prixfixe.prixfixe import PrixFixeNet

TRAIN_DATA_PATH = "/scratch/st-cdeboer-1/sambina/mpra/mpra_models/random-promoter-dream-challenge-2022/data/demo_train.txt" 
VALID_DATA_PATH = "/scratch/st-cdeboer-1/sambina/mpra/mpra_models/random-promoter-dream-challenge-2022/data/demo_val.txt" 
TRAIN_BATCH_SIZE = 512 
BATCH_PER_EPOCH = 10 
N_PROCS = 8
VALID_BATCH_SIZE = 4096
BATCH_PER_VALIDATION = 10
PLASMID_PATH = "/scratch/st-cdeboer-1/sambina/mpra/mpra_models/random-promoter-dream-challenge-2022/data/plasmid.json"
SEQ_SIZE = 150
NUM_EPOCHS = 5 
CUDA_DEVICE_ID = 0
lr = 0.005 

ref_df = pd.read_csv("/scratch/st-cdeboer-1/sambina/position_variant_effect/paper_analysis/figure_3/yeast_motifs/output/all_yeast_variants_full_knockout.csv")

def predict(df, col):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    generator = torch.Generator()
    generator.manual_seed(2147483647)
    
    dataprocessor = AutosomeDataProcessor(
        path_to_training_data=TRAIN_DATA_PATH,
        path_to_validation_data=VALID_DATA_PATH,
        train_batch_size=TRAIN_BATCH_SIZE, 
        batch_per_epoch=BATCH_PER_EPOCH,
        train_workers=N_PROCS,
        valid_batch_size=VALID_BATCH_SIZE,
        valid_workers=N_PROCS,
        shuffle_train=True,
        shuffle_val=False,
        plasmid_path=PLASMID_PATH,
        seqsize=SEQ_SIZE,
        generator=generator
    )

    first = BHIFirstLayersBlock(
        in_channels=dataprocessor.data_channels(),
        out_channels=320,
        seqsize=dataprocessor.data_seqsize(),
        kernel_sizes=[9, 15],
        pool_size=1,
        dropout=0.2
    )

    core = AutosomeCoreBlock(
        in_channels=first.out_channels,
        out_channels=64,
        seqsize=first.infer_outseqsize()
    )

    final = AutosomeFinalLayersBlock(
        in_channels=core.out_channels, 
        seqsize=core.infer_outseqsize()
    )

    model = PrixFixeNet(
        first=first,
        core=core,
        final=final,
        generator=generator
    )

    # Move model to the appropriate device (GPU or CPU)
    model.to(device)

    predictor = AutosomePredictor(
        model=model, 
        model_pth='/scratch/st-cdeboer-1/sambina/position_variant_effect/paper_analysis/figure_3/yeast_motifs/input/dream-cnn.pth',  
        device=device  
    )
    
    tqdm.pandas()
    
    # Apply prediction to dataframe
    df[f'pred_{col}'] = df[col].progress_apply(lambda seq: predictor.predict(seq))
    print(df.head())
    return df

offsets = list(range(-30, 31, 1))
for offset in offsets:
    result_ref = predict(ref_df, f"seq_{offset}")
  
result_ref.to_csv("/scratch/st-cdeboer-1/sambina/position_variant_effect/paper_analysis/figure_3/yeast_motifs/output/predicted_all_yeast_full_knockout.csv")
