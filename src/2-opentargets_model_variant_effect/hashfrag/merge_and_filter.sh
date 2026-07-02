#!/bin/bash
#SBATCH --job-name=hashfrag_merge_filter
#SBATCH --account=st-cdeboer-1
#SBATCH --output=/scratch/st-cdeboer-1/sambina/outputs/%A:%x.txt
#SBATCH --error=/scratch/st-cdeboer-1/sambina/errors/%A:%x.err
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=skylake
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sambina.aninta@ubc.ca

source ~/.bashrc
conda activate hashFrag

HASHFRAG_OUT=/scratch/st-cdeboer-1/sambina/position_mpra/outputs/2-opentargets_model_variant_effect/hashfrag
SCRIPTS=/scratch/st-cdeboer-1/sambina/position_mpra/src/2-opentargets_model_variant_effect/hashfrag

echo "[$(date)] Merging similar_pairs.tsv from all 100 chunks..."
cat ${HASHFRAG_OUT}/chunk_*/hashFrag.similar_pairs.tsv > ${HASHFRAG_OUT}/hashFrag.similar_pairs.tsv
echo "[$(date)] Merged $(wc -l < ${HASHFRAG_OUT}/hashFrag.similar_pairs.tsv) total pairs."

echo "[$(date)] Expanding clashes and filtering full offsets FASTA..."
python ${SCRIPTS}/apply_hashfrag_filter.py
echo "[$(date)] Done."
