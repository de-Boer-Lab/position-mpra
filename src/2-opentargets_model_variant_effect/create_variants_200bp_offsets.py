from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import pandas as pd

OUTPUT_FA = "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/2-opentargets_model_variant_effect/variants_200bp_offsets.fa"


def main():
    snvs_only = pd.read_csv(
        "/scratch/st-cdeboer-1/sambina/position_mpra/data/opentargets_model_data/snvs_only_gwas_0.1_postprob.csv"
    )
    genome_path = "/scratch/st-cdeboer-1/sambina/reference_genome/hg38.fa"

    print("Loading genome...")
    with open(genome_path, "rt") as handle:
        genome = SeqIO.to_dict(SeqIO.parse(handle, "fasta"))

    sidemer = 100
    offsets = [-90, -80, -60, -40, -20, 0, 20, 40, 60, 80, 90]

    def generate_sequences(row, sidemer_offset=0):
        chrom = "chr" + str(row["CHR_ID"])
        pos = int(row["Position"]) - 1
        ref_nucleotide = row["Ref"]
        alt_nucleotide = row["Alt"]

        sequence_upstream = genome[chrom].seq[pos - sidemer + sidemer_offset : pos].upper()
        sequence_downstream = genome[chrom].seq[pos + 1 : pos + sidemer + sidemer_offset].upper()

        ref_sequence = str(sequence_upstream) + ref_nucleotide + str(sequence_downstream)
        alt_sequence = str(sequence_upstream) + alt_nucleotide + str(sequence_downstream)

        if ref_sequence == alt_sequence:
            print(
                f"Assertion failed at position {row['CHR_ID']}:{row['Position']} with ref {row['Ref']} and alt {row['Alt']}"
            )
            assert ref_sequence != alt_sequence

        return ref_sequence, alt_sequence

    records = []
    for offset in offsets:
        print(f"Processing offset {offset}...")
        for _, row in snvs_only.iterrows():
            variant_id = f"chr{row['CHR_ID']}_{row['Position']}_{row['Ref']}_{row['Alt']}"
            ref_seq, alt_seq = generate_sequences(row, sidemer_offset=offset)
            records.append(
                SeqRecord(Seq(ref_seq), id=f"{variant_id}_offset{offset}_ref", description="")
            )
            records.append(
                SeqRecord(Seq(alt_seq), id=f"{variant_id}_offset{offset}_alt", description="")
            )

    print(f"Writing {len(records)} sequences to {OUTPUT_FA}...")
    with open(OUTPUT_FA, "w") as out_handle:
        SeqIO.write(records, out_handle, "fasta")
    print("Done.")


if __name__ == "__main__":
    main()
