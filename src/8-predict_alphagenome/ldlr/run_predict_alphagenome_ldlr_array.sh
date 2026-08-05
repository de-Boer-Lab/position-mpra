#!/bin/bash
#SBATCH --job-name=predict_alphagenome_ldlr
#SBATCH --account=st-cdeboer-1-gpu
#SBATCH --output=/scratch/st-cdeboer-1/sambina/outputs/%A_%a:%x.txt
#SBATCH --error=/scratch/st-cdeboer-1/sambina/errors/%A_%a:%x.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-19
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sambina.aninta@ubc.ca

# Compute phase: run AFTER run_predict_alphagenome_ldlr_setup.sh has finished --
# this array's tasks only open ref.npy/alt.npy/done_mask.npy in mode="r+" (never
# create them), so they need those files to already exist. Each of the 20 tasks
# takes a disjoint slice of the ~1,639-window K-sweep, for both LDLR variants.
#
# NUM_TASKS below must match the --array range above (0-19 = 20 tasks total).
# If you change one, change the other. The %4 throttle caps it at 4 running
# concurrently (matches one node's GPU count on the "gpu" partition) -- raise
# or drop it if you know you have more V100s available to your account at once.
#
# --time is a PER-TASK limit, not a budget for the whole array: each task only
# has to finish its own ~164 windows (82/variant x 2 variants), regardless of the
# %4 throttle. At ~1 min/window that's ~2h44m; 06:00:00 pads for model-load
# overhead and the fact that 1 min/window is still an untested guess -- run the
# smoke-test instructions in predict_alphagenome_ldlr.py's docstring first and
# re-adjust once you have real per-window timing.
#
# The %4 throttle does NOT change any single task's required --time -- it only
# changes how long the ARRAY AS A WHOLE takes to drain all 20 tasks, since only
# 4 run at once and the rest queue. With no throttle, all 20 finish in parallel
# in ~2h44m total. With %4, they run in 5 sequential waves of 4, so the last
# task doesn't even start until ~3 waves (~8h) in -- total array wall-clock is
# ~5 x 2h44m =~13.7h, even though each individual task's own run is still short.

NUM_TASKS=20

source ~/.bashrc
conda activate alphagenome_pt

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

SCRIPT=/scratch/st-cdeboer-1/sambina/position_mpra/src/8-predict_alphagenome/ldlr/predict_alphagenome_ldlr.py

python "$SCRIPT" --task_id "$SLURM_ARRAY_TASK_ID" --num_tasks "$NUM_TASKS" --device cuda
