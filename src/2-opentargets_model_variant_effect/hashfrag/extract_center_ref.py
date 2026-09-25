"""
Extract only the offset-0 ref sequence for each variant.
This reduces the hashFrag input from 2.27M → 103K sequences (22x smaller).
After hashFrag runs, use apply_hashfrag_filter.py to expand clashes back to
all offsets for both ref and alt.
"""

from pathlib import Path
from Bio import SeqIO

input_fa = Path(
    "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/2-opentargets_model_variant_effect/variants_200bp_offsets.fa"
)
output_fa = Path(
    "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/2-opentargets_model_variant_effect/variants_200bp_offset0_ref.fa"
)

records = (rec for rec in SeqIO.parse(input_fa, "fasta") if rec.id.endswith("_offset0_ref"))

count = SeqIO.write(records, output_fa, "fasta")
print(f"Wrote {count} sequences to {output_fa}")
