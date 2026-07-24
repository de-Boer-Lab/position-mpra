#!/usr/bin/env bash
set -euo pipefail

BEDTOOLS=/arc/project/st-cdeboer-1/sambina/miniconda3/envs/chrombpnet_copy/bin/bedtools
CHROM_SIZES=/scratch/st-cdeboer-1/sambina/reference_genome/hg38.chrom.sizes

OUT_DIR=/scratch/st-cdeboer-1/sambina/position_mpra/outputs/8-aphagenome/all_k562_promoters
RAW_DIR="$OUT_DIR/raw"
PROC_DIR="$OUT_DIR/processed"

TSS_BED=$(ls "$PROC_DIR"/gencode.v*.tss.bed)
ATAC_PEAKS="$RAW_DIR/k562_atac_idr_peaks.GRCh38.bed.gz"

FLANK=500

TSS_SORTED="$PROC_DIR/tss.sorted.bed"
ATAC_SORTED="$PROC_DIR/k562_atac.sorted.bed"
TSS_FLANK="$PROC_DIR/tss.flank${FLANK}.sorted.bed"

sort -k1,1 -k2,2n "$TSS_BED" > "$TSS_SORTED"
zcat "$ATAC_PEAKS" | sort -k1,1 -k2,2n > "$ATAC_SORTED"

"$BEDTOOLS" slop -i "$TSS_SORTED" -g "$CHROM_SIZES" -b "$FLANK" > "$TSS_FLANK"

"$BEDTOOLS" intersect -a "$TSS_FLANK" -b "$ATAC_SORTED" -u \
    > "$OUT_DIR/k562_promoters_in_open_chromatin.bed"

N_TSS=$(wc -l < "$TSS_SORTED")
N_OPEN=$(wc -l < "$OUT_DIR/k562_promoters_in_open_chromatin.bed")
echo "Total transcript TSS: $N_TSS"
echo "TSS (+/-${FLANK}bp) overlapping K562 ATAC peaks: $N_OPEN"
