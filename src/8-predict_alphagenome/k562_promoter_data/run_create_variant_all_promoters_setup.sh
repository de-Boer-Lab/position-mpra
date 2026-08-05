#!/bin/bash
#SBATCH --job-name=create_variant_all_promoters_setup
#SBATCH --account=st-cdeboer-1
#SBATCH --output=/scratch/st-cdeboer-1/sambina/outputs/%A:%x.txt
#SBATCH --error=/scratch/st-cdeboer-1/sambina/errors/%A:%x.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sambina.aninta@ubc.ca

# One-time setup: build promoters_metadata.tsv (~17,535 genes after the
# chromosome-boundary and exon2-reach filters) and preallocate
# {baseline,upstream,downstream}_{ref,alt}.npy + done_mask.npy. No GPU
# needed. Must finish before submitting run_create_variant_all_promoters_array.sh
# -- the array job only opens these files in mode="r+".

source ~/.bashrc
conda activate alphagenome_pt

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

SCRIPT=/scratch/st-cdeboer-1/sambina/position_mpra/src/8-predict_alphagenome/k562_promoter_data/create_variant_all_promoters.py

python "$SCRIPT" --setup_only
