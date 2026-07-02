#!/bin/bash
#SBATCH --job-name=predict_alphagenome
#SBATCH --account=st-cdeboer-1
#SBATCH --output=/scratch/st-cdeboer-1/sambina/outputs/%A:%x.txt
#SBATCH --error=/scratch/st-cdeboer-1/sambina/errors/%A:%x.err
#SBATCH --time=80:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sambina.aninta@ubc.ca

source ~/.bashrc
conda activate alphagenome_pt

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

SCRIPT=/scratch/st-cdeboer-1/sambina/position_mpra/src/8-predict_alphagenome/pytorch/predict_alphagenome_pt.py

python "$SCRIPT" --chrom chr5 --var_pos 1295113 --ref G --alt A --name C228T
python "$SCRIPT" --chrom chr5 --var_pos 1295135 --ref G --alt A --name C250T