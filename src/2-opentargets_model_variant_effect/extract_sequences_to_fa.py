from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import pandas as pd

INPUT = "/scratch/st-cdeboer-1/sambina/mpra/data/chromosome/gosai/data_lfcse/data_k562/fold_4/train.txt.gz"
OUTPUT = (
    "/scratch/st-cdeboer-1/sambina/mpra/data/chromosome/gosai/data_lfcse/data_k562/fold_4/train.fa"
)

TRIM_START = 15
TRIM_END = 16
TARGET_LEN = 200

print(f"Reading {INPUT}...")
df = pd.read_csv(INPUT, sep="\t", compression="gzip", usecols=["seq_id", "seq"])

records = []
for _, row in df.iterrows():
    seq = row["seq"]
    if len(seq) > TARGET_LEN:
        seq = seq[TRIM_START : len(seq) - TRIM_END]
    records.append(SeqRecord(Seq(seq), id=str(row["seq_id"]), description=""))

print(f"Writing {len(records)} sequences to {OUTPUT}...")
with open(OUTPUT, "w") as out_handle:
    SeqIO.write(records, out_handle, "fasta")

print("Done.")
