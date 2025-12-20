#!/bin/bash
#SBATCH --job-name=run_ism  
#SBATCH --account=st-cdeboer-1-gpu
#SBATCH --output=/scratch/st-cdeboer-1/sambina/outputs/%A:%x.txt
#SBATCH --error=/scratch/st-cdeboer-1/sambina/errors/%A:%x.err
#SBATCH --time=168:00:00                    
#SBATCH --gres=gpu:1        
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8                  
#SBATCH --mem=64G                          
#SBATCH --mail-type=ALL                    
#SBATCH --mail-user=sambina.aninta@ubc.ca  

source ~/.bashrc
conda activate dream_rocky_3

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python /scratch/st-cdeboer-1/sambina/position_mpra/src/6-ism_variant_effects/predict_ism_for_all_OT.py