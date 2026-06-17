#!/bin/bash
#SBATCH --job-name=hashfrag_filter_test_set
#SBATCH --account=st-cdeboer-1
#SBATCH --output=/scratch/st-cdeboer-1/sambina/outputs/%A:%x.txt
#SBATCH --error=/scratch/st-cdeboer-1/sambina/errors/%A:%x.err
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=skylake
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sambina.aninta@ubc.ca

source ~/.bashrc
conda activate hashFrag

# Install hashFrag CLI if not already installed

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

SCRIPTS=/scratch/st-cdeboer-1/sambina/position_mpra/src/2-opentargets_model_variant_effect/hashfrag

# Step 1: run hashFrag on all 2.27M sequences (all offsets, ref + alt)
hashFrag filter_existing_splits \
    --train-fasta-path /scratch/st-cdeboer-1/sambina/mpra/data/chromosome/gosai/data_lfcse/data_k562/fold_4/train.fa \
    --test-fasta-path /scratch/st-cdeboer-1/sambina/position_mpra/outputs/2-opentargets_model_variant_effect/variants_200bp_offsets.fa \
    -t 60 \
    -o /scratch/st-cdeboer-1/sambina/position_mpra/outputs/2-opentargets_model_variant_effect/hashfrag

# Step 2: if any offset (ref or alt) clashes, remove all offsets for both ref and alt
python ${SCRIPTS}/apply_hashfrag_filter.py
