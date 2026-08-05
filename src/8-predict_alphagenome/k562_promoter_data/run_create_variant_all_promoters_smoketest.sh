#!/bin/bash
#SBATCH --job-name=create_variant_all_promoters_smoketest
#SBATCH --account=st-cdeboer-1-gpu
#SBATCH --output=/scratch/st-cdeboer-1/sambina/outputs/%A:%x.txt
#SBATCH --error=/scratch/st-cdeboer-1/sambina/errors/%A:%x.err
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sambina.aninta@ubc.ca

# Smoke test only -- 3 genes, GPU, isolated --out_dir so it never touches the
# real production array files (which will be sized for all ~17,535 genes).
# Purpose: validate the actual model forward pass end-to-end (coordinate/
# metadata logic was already validated on CPU without the model).

source ~/.bashrc
conda activate alphagenome_pt

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

SCRIPT=/scratch/st-cdeboer-1/sambina/position_mpra/src/8-predict_alphagenome/k562_promoter_data/create_variant_all_promoters.py
SMOKE_OUT=/scratch/st-cdeboer-1/sambina/position_mpra/outputs/8-aphagenome/all_k562_promoters/predictions_smoketest

python "$SCRIPT" --max_genes 3 --device cuda --out_dir "$SMOKE_OUT"
