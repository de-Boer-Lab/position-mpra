#!/bin/bash
#SBATCH --job-name=opentarget_prediction       
#SBATCH --account=st-cdeboer-1-gpu
#SBATCH --output=/scratch/st-cdeboer-1/sambina/outputs/%A:%x.txt
#SBATCH --error=/scratch/st-cdeboer-1/sambina/errors/%A:%x.err
#SBATCH --time=40:00:00                    
#SBATCH --gres=gpu:1        
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8                  
#SBATCH --mem=64G                          
#SBATCH --mail-type=ALL                    
#SBATCH --mail-user=sambina.aninta@ubc.ca  

source ~/.bashrc
conda activate dream_rocky_2

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

BASE_SCRIPT_PATH="/scratch/st-cdeboer-1/sambina/position_mpra/experiments/opentargets_model_variant_effect/open_targets_offset_gosai_revcomp.py"
BASE_MODEL_DIR="/scratch/st-cdeboer-1/sambina"
BASE_OUTPUT_DIR="/scratch/st-cdeboer-1/sambina/position_mpra/outputs/opentargets_model/gosai"

declare -A models=(
    ["k562_gosai"]="/scratch/st-cdeboer-1/sambina/mpra/mpra_with_chromosome/gosai_2024/output_lfcse/output_k562/fold_4/model_best.pth"
)


for key in "${!models[@]}"; do
    model_path="${models[$key]}"
    output_path="$BASE_OUTPUT_DIR/${key}_revcomp.csv.gz"
    echo "Running job for model: $model_path, output: $output_path"
    srun --exclusive --gres=gpu:1 python $BASE_SCRIPT_PATH --model_path $model_path --output_path $output_path &
done

wait
