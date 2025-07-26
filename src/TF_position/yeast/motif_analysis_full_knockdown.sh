#!/bin/bash
#SBATCH --job-name=motif_yeast_kc
#SBATCH --account=st-cdeboer-1-gpu
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90GB
#SBATCH --output=/scratch/st-cdeboer-1/sambina/outputs/%A:%x.txt
#SBATCH --error=/scratch/st-cdeboer-1/sambina/errors/%A:%x.err
#SBATCH --mail-user=sambina.aninta@ubc.ca
#SBATCH --mail-type=ALL

source ~/.bashrc 
conda activate dream_rocky

# Update LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH=/arc/project/st-cdeboer-1/sambina/miniconda3/envs/dream_rocky/bin:$PATH
python /scratch/st-cdeboer-1/sambina/position_variant_effect/paper_analysis/figure_3/yeast_motifs/analysis/random_sequence_ism_average_full_knockdown.py &
python /scratch/st-cdeboer-1/sambina/position_variant_effect/paper_analysis/figure_3/yeast_motifs/analysis/random_sequence_ism_average_full_knockdown_rc.py &

wait