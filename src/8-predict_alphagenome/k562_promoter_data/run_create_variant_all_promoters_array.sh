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
#SBATCH --array=0-59%12
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sambina.aninta@ubc.ca

# Compute phase: run AFTER run_create_variant_all_promoters_setup.sh has
# finished -- this array's tasks only open the *_ref.npy/*_alt.npy/done_mask.npy
# files in mode="r+" (never create them).
#
# NUM_TASKS below must match the --array range above (0-59 = 60 tasks total).
#
# create_variant_all_promoters.py now tests every gene at all 4 insertion
# lengths at both positions (9 conditions/gene, was 3 -- see that script's
# docstring), so per-gene compute is ~3x what it was. To compensate:
#   - NUM_TASKS tripled (20 -> 60), so genes/task drops ~3x (~877 -> ~292)
#     and per-task compute time is back to roughly what it was before
#     (~292 genes/task * ~12.9s/gene [9 conditions] ~= same ~63 min/task as
#     the old ~877 genes/task * ~4.3s/gene [3 conditions]) -- --time=02:00:00
#     still pads generously, no change needed there.
#   - Throttle tripled (%4 -> %12) so total wall-clock across the array stays
#     close to the original ~5-6h estimate (~7h observed) instead of tripling
#     to ~15-21h: 60 tasks / 12 concurrent = 5 waves, same as 20/4 before.
#   %12 assumes your account's GPU allocation can actually run 12 concurrent
#   gres=gpu:1 jobs at once -- if your fair-share/QOS caps it lower, SLURM
#   will just run fewer at a time (more waves, longer wall-clock) without
#   any script changes needed; lower %12 to match your quota if jobs are
#   queuing instead of running. Check with `sacctmgr show assoc user=$USER`
#   or just watch `squeue -u $USER` after submitting.

NUM_TASKS=60

source ~/.bashrc
conda activate alphagenome_pt

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

SCRIPT=/scratch/st-cdeboer-1/sambina/position_mpra/src/8-predict_alphagenome/k562_promoter_data/create_variant_all_promoters.py

python "$SCRIPT" --task_id "$SLURM_ARRAY_TASK_ID" --num_tasks "$NUM_TASKS" --device cuda
