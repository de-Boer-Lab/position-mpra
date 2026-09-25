#!/bin/bash
#SBATCH --job-name=opentarget_prediction       
#SBATCH --account=st-cdeboer-1-gpu
#SBATCH --output=/scratch/st-cdeboer-1/sambina/outputs/%A:%x.txt
#SBATCH --error=/scratch/st-cdeboer-1/sambina/errors/%A:%x.err
#SBATCH --time=80:00:00                    
#SBATCH --gres=gpu:1        
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8                  
#SBATCH --mem=64G                          
#SBATCH --mail-type=ALL                    
#SBATCH --mail-user=sambina.aninta@ubc.ca  

source ~/.bashrc
conda activate dream_rocky_3
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

BASE_SCRIPT_PATH="/scratch/st-cdeboer-1/sambina/position_mpra/src/2-opentargets_model_variant_effect/hashfrag/predict_targets_offset_gosai.py"
BASE_OUTPUT_DIR="/scratch/st-cdeboer-1/sambina/position_mpra/outputs/2-opentargets_model_variant_effect/hashfrag/gosai"
mkdir -p "$BASE_OUTPUT_DIR"

model_path="/scratch/st-cdeboer-1/sambina/mpra/output/chromosome/gosai/output_lfcse/output_k562/fold_4/model_best.pth"
output_path="$BASE_OUTPUT_DIR/k562_gosai_filtered.csv.gz"

python $BASE_SCRIPT_PATH --model_path "$model_path" --output_path "$output_path"