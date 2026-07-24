#!/bin/bash
#SBATCH --job-name=create_variant_all_promoters
#SBATCH --account=st-cdeboer-1-gpu
#SBATCH --output=/scratch/st-cdeboer-1/sambina/outputs/%A_%a:%x.txt
#SBATCH --error=/scratch/st-cdeboer-1/sambina/errors/%A_%a:%x.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-19%4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sambina.aninta@ubc.ca

# Compute phase: run AFTER run_create_variant_all_promoters_setup.sh has
# finished -- this array's tasks only open the *_ref.npy/*_alt.npy/done_mask.npy
# files in mode="r+" (never create them).
#
# NUM_TASKS below must match the --array range above (0-19 = 20 tasks total).
# %4 throttles to 4 concurrent (matches predict_alphagenome_ldlr's convention).
#
# Timing: the 3-gene GPU smoke test (job 12349892) measured ~4.3s/gene for
# all 3 conditions combined (after ~15s one-time model-load/warmup). 17,535
# genes / 20 tasks = ~877 genes/task -> ~63 min/task at that rate.
# --time=02:00:00 pads generously for variance. With %4, the whole array
# (5 sequential waves of 4) finishes in ~5-6h wall-clock.

NUM_TASKS=20

source ~/.bashrc
conda activate alphagenome_pt

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

SCRIPT=/scratch/st-cdeboer-1/sambina/position_mpra/src/8-predict_alphagenome/k562_promoter_data/create_variant_all_promoters.py

python "$SCRIPT" --task_id "$SLURM_ARRAY_TASK_ID" --num_tasks "$NUM_TASKS" --device cuda
