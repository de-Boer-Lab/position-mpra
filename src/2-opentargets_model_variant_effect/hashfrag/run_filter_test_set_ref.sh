#!/bin/bash
#SBATCH --job-name=hashfrag_filter_test_set
#SBATCH --account=st-cdeboer-1
#SBATCH --output=/scratch/st-cdeboer-1/sambina/outputs/%A_%a:%x.txt
#SBATCH --error=/scratch/st-cdeboer-1/sambina/errors/%A_%a:%x.err
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=1-100
#SBATCH --partition=skylake
#SBATCH --mem=32G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=sambina.aninta@ubc.ca

source ~/.bashrc
conda activate hashFrag
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

CHUNK_ID=${SLURM_ARRAY_TASK_ID}
CHUNK_DIR=/scratch/st-cdeboer-1/sambina/position_mpra/outputs/2-opentargets_model_variant_effect/hashfrag/chunks
OUT_DIR=/scratch/st-cdeboer-1/sambina/position_mpra/outputs/2-opentargets_model_variant_effect/hashfrag/chunk_${CHUNK_ID}

mkdir -p "${OUT_DIR}"

echo "[$(date)] Task ${CHUNK_ID}/100: running hashFrag on chunk_${CHUNK_ID}.fa..."
hashFrag filter_existing_splits \
    --train-fasta-path /scratch/st-cdeboer-1/sambina/mpra/data/chromosome/gosai/data_lfcse/data_k562/fold_4/train.fa \
    --test-fasta-path ${CHUNK_DIR}/chunk_${CHUNK_ID}.fa \
    -t 8 \
    -o ${OUT_DIR}
echo "[$(date)] Task ${CHUNK_ID}/100: done."
