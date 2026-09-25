"""
After running hashFrag on the offset-0 ref sequences, expand clashes to
remove all offsets for both ref and alt of any affected variant.

Input:
  - hashFrag.similar_pairs.tsv  (id_i, id_j, score — no header)
  - variants_200bp_offsets.fa   (full 2.27M sequence FASTA)

Output:
  - variants_200bp_offsets_hashfrag_filtered.fa
"""

import re
from pathlib import Path

import pandas as pd
from Bio import SeqIO

HASHFRAG_DIR = Path(
    "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/2-opentargets_model_variant_effect/hashfrag"
)
FULL_FA = Path(
    "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/2-opentargets_model_variant_effect/variants_200bp_offsets.fa"
)
OUT_FA = FULL_FA.parent / "variants_200bp_offsets_hashfrag_filtered.fa"

hits = pd.read_csv(
    HASHFRAG_DIR / "hashFrag.similar_pairs.tsv",
    sep="\t",
    header=None,
    names=["id_i", "id_j", "score"],
)

# id_i is the test sequence; extract variant (strip _offset*_ref/alt suffix)
clashing_variants = hits["id_i"].str.replace(r"_offset-?\d+_(ref|alt)$", "", regex=True).unique()
clashing_set = set(clashing_variants)
print(f"Variants with at least one clashing offset: {len(clashing_set)}")


def variant_id(seq_id: str) -> str:
    return re.sub(r"_offset-?\d+_(ref|alt)$", "", seq_id)


kept, removed = 0, 0
with open(OUT_FA, "w") as out:
    for rec in SeqIO.parse(FULL_FA, "fasta"):
        if variant_id(rec.id) in clashing_set:
            removed += 1
        else:
            SeqIO.write(rec, out, "fasta")
            kept += 1

print(f"Removed: {removed} sequences")
print(f"Kept:    {kept} sequences")
print(f"Written to {OUT_FA}")
