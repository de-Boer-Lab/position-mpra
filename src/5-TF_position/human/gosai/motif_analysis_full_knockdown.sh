#!/bin/bash
#SBATCH --job-name=motif_predict_full_knockout_human
#SBATCH --account=st-cdeboer-1-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90GB
#SBATCH --output=/scratch/st-cdeboer-1/sambina/outputs/%A:%x.txt
#SBATCH --error=/scratch/st-cdeboer-1/sambina/errors/%A:%x.err
#SBATCH --mail-user=sambina.aninta@ubc.ca
#SBATCH --mail-type=ALL

source ~/.bashrc 
conda activate dream_rocky_3

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH=/arc/project/st-cdeboer-1/sambina/miniconda3/envs/dream_rocky_3/bin:$PATH
python /scratch/st-cdeboer-1/sambina/position_mpra/src/5-TF_position/human/gosai/random_sequence_knockdown_rc.py &
python /scratch/st-cdeboer-1/sambina/position_mpra/src/5-TF_position/human/gosai/random_sequence_knockdown.py &

wait