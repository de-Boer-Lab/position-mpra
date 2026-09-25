#!/bin/bash
#SBATCH --job-name=predict_alphagenome_ldlr_setup
#SBATCH --account=st-cdeboer-1
#SBATCH --output=/scratch/st-cdeboer-1/sambina/outputs/%A:%x.txt
#SBATCH --error=/scratch/st-cdeboer-1/sambina/errors/%A:%x.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sambina.aninta@ubc.ca

# One-time setup: preallocate ref.npy/alt.npy/done_mask.npy + write meta.tsv for
# both LDLR variants. No GPU needed. Must finish (check sacct/squeue) before
# submitting run_predict_alphagenome_ldlr_array.sh -- the array job only opens
# these files in mode="r+" and will error out if they don't exist yet.

source ~/.bashrc
conda activate alphagenome_pt

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

SCRIPT=/scratch/st-cdeboer-1/sambina/position_mpra/src/8-predict_alphagenome/ldlr/predict_alphagenome_ldlr.py

python "$SCRIPT" --setup_only
