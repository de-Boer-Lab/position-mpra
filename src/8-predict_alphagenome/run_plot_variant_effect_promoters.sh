#!/bin/bash
#SBATCH --job-name=plot_variant_effect_promoters
#SBATCH --account=st-cdeboer-1
#SBATCH --output=/scratch/st-cdeboer-1/sambina/outputs/%A:%x.txt
#SBATCH --error=/scratch/st-cdeboer-1/sambina/errors/%A:%x.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sambina.aninta@ubc.ca

# Post-processing only -- CPU, no GPU needed. Run AFTER the array job
# (run_create_variant_all_promoters_array.sh) has fully finished, i.e. once
# done_mask.npy is all True for every gene (17,535 x 3 conditions).
#
# To chain it automatically instead of running it manually, submit with:
#   sbatch --dependency=afterok:<array_job_id> run_plot_variant_effect_promoters.sh
# (<array_job_id> is the numeric job ID sbatch printed for the array job --
# --dependency=afterok waits for ALL array tasks to finish successfully.)
#
# Reads {baseline,upstream,downstream}_{ref,alt}.npy + promoters_metadata.tsv,
# computes exon2 variant effect per gene per condition, and writes:
#   exon2_variant_effect.tsv   -- tidy long-format table (gene, condition, length, exon2_ve)
#   exon2_variant_effect_boxplot.png
#   exon2_variant_effect_heatmap.png -- one heatmap per insertion length (25/50/75/100bp),
#     rows = genes, columns = no insertion / random at -100 / random at -200

source ~/.bashrc
conda activate dream_rocky_3

SCRIPT=/scratch/st-cdeboer-1/sambina/position_mpra/src/8-predict_alphagenome/k562_promoter_data/plot_variant_effect_promoters.py

python "$SCRIPT"
