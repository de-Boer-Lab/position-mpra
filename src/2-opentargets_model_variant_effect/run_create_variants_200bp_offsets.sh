#!/bin/bash
#SBATCH --job-name=create_variants_200bp_offsets
#SBATCH --account=st-cdeboer-1
#SBATCH --output=/scratch/st-cdeboer-1/sambina/outputs/%A:%x.txt
#SBATCH --error=/scratch/st-cdeboer-1/sambina/errors/%A:%x.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sambina.aninta@ubc.ca

source ~/.bashrc
conda activate dream_rocky_3

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python /scratch/st-cdeboer-1/sambina/position_mpra/src/2-opentargets_model_variant_effect/create_variants_200bp_offsets.py
