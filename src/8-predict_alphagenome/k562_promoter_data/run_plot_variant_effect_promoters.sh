#!/bin/bash
#SBATCH --job-name=plot_variant_effect_promoters
#SBATCH --account=st-cdeboer-1
#SBATCH --output=/scratch/st-cdeboer-1/sambina/outputs/%A:%x.txt
#SBATCH --error=/scratch/st-cdeboer-1/sambina/errors/%A:%x.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sambina.aninta@ubc.ca

# Post-processing only -- CPU, no GPU needed. Run AFTER the array job
# (run_create_variant_all_promoters_array.sh) has fully finished, i.e. once
# done_mask.npy is all True for every gene (17,535 x 9 conditions).
#
# To chain it automatically instead of running it manually, submit with:
#   sbatch --dependency=afterok:<array_job_id> run_plot_variant_effect_promoters.sh
# (<array_job_id> is the numeric job ID sbatch printed for the array job --
# --dependency=afterok waits for ALL array tasks to finish successfully.)
#
# compute_exon2_ve is parallelized across a process pool (one chunk of genes
# per condition per worker) -- a prior single-threaded run measured ~20
# rows/sec/condition (~17,535 genes x 9 conditions sequentially -> ~2h+).
# --cpus-per-task=32 gives that pool 32 workers instead of 1, which should
# land VE computation in single-digit minutes assuming reasonable scaling
# (network-filesystem I/O latency is what parallelizes well here -- lower
# this if your account's CPU quota can't sustain 32 cores on one job, or if
# you observe heavy I/O contention with many concurrent readers).
#
# Reads {condition_key}_{ref,alt}.npy (CONDITION_KEYS in
# create_variant_all_promoters.py) + promoters_metadata.tsv, computes exon2
# variant effect per gene per condition, and writes (all figures as .svg):
#   exon2_variant_effect.tsv   -- tidy long-format table (gene, condition, length, exon2_ve)
#   exon2_variant_effect_boxplot.svg
#   exon2_variant_effect_heatmap.svg -- one heatmap per insertion length (25/50/75/100bp),
#     rows = genes, columns = no insertion / random at -100 / random at -200
#   exon2_ve_correlation_heatmap.svg / _upstream.svg -- pairwise Pearson R^2 heatmaps
#   exon2_ve_spearman_heatmap.svg / _upstream.svg -- pairwise Spearman R^2 heatmaps
#   exon2_ve_scatter_matrix.svg / _upstream.svg -- all-by-all scatterplot matrices
#   exon2_ve_scatter_grid.svg -- 3x3 baseline-vs-condition scatterplots

source ~/.bashrc
conda activate dream_rocky_3

SCRIPT=/scratch/st-cdeboer-1/sambina/position_mpra/src/8-predict_alphagenome/k562_promoter_data/plot_variant_effect_promoters.py

python "$SCRIPT" --workers "$SLURM_CPUS_PER_TASK"
