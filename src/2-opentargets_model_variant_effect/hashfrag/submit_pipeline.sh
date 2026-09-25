#!/bin/bash
# Splits the test FASTA into 100 chunks, submits the array job,
# then submits the merge+filter job to run after all array tasks finish.


TEST_FA=/scratch/st-cdeboer-1/sambina/position_mpra/outputs/2-opentargets_model_variant_effect/variants_200bp_offset0_ref.fa
HASHFRAG_OUT=/scratch/st-cdeboer-1/sambina/position_mpra/outputs/2-opentargets_model_variant_effect/hashfrag
CHUNK_DIR=${HASHFRAG_OUT}/chunks
SCRIPTS=/scratch/st-cdeboer-1/sambina/position_mpra/src/2-opentargets_model_variant_effect/hashfrag

mkdir -p "${CHUNK_DIR}"

# --- Split FASTA into 100 chunks ---
N=$(grep -c "^>" "${TEST_FA}")
CHUNK_SIZE=$(( (N + 99) / 100 ))
echo "Total sequences: ${N}, ~${CHUNK_SIZE} per chunk"

python3 - <<EOF
import math

fa = "${TEST_FA}"
chunk_dir = "${CHUNK_DIR}"
chunk_size = ${CHUNK_SIZE}

seq_count = 0
current_chunk = 0
out = None

with open(fa) as f:
    for line in f:
        if line.startswith(">"):
            seq_count += 1
            new_chunk = (seq_count - 1) // chunk_size + 1
            if new_chunk != current_chunk:
                if out:
                    out.close()
                out = open(f"{chunk_dir}/chunk_{new_chunk}.fa", "w")
                current_chunk = new_chunk
        out.write(line)

if out:
    out.close()
EOF

echo "FASTA split into $(ls ${CHUNK_DIR}/chunk_*.fa | wc -l) chunks."

# --- Submit array job ---
ARRAY_JOB_ID=$(sbatch --parsable "${SCRIPTS}/run_filter_test_set_ref.sh")
echo "Array job submitted: ${ARRAY_JOB_ID}"

# --- Submit merge+filter job, runs only after all array tasks succeed ---
MERGE_JOB_ID=$(sbatch --parsable --dependency=afterok:${ARRAY_JOB_ID} "${SCRIPTS}/merge_and_filter.sh")
echo "Merge+filter job submitted: ${MERGE_JOB_ID} (depends on ${ARRAY_JOB_ID})"

echo ""
echo "Monitor with: squeue -u \$USER"
