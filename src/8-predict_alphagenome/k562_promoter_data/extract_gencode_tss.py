"""
Download the latest GENCODE human GTF (hg38) and extract a strand-aware
TSS bed file (one row per transcript): chrom, tss-1, tss, gene_name,
transcript_id, strand.
"""

import gzip
import pathlib
import re
import requests

RAW_DIR = pathlib.Path(
    "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/8-aphagenome/all_k562_promoters/raw"
)
PROCESSED_DIR = pathlib.Path(
    "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/8-aphagenome/all_k562_promoters/processed"
)
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INDEX_URL = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/"
r = requests.get(INDEX_URL)
r.raise_for_status()
releases = sorted(int(m) for m in re.findall(r"release_(\d+)/", r.text))
latest = releases[-1]
print(f"Latest GENCODE human release: {latest}")

GTF_URL = f"{INDEX_URL}release_{latest}/gencode.v{latest}.annotation.gtf.gz"
GTF_PATH = RAW_DIR / f"gencode.v{latest}.annotation.gtf.gz"

if not GTF_PATH.exists():
    r = requests.get(GTF_URL, stream=True)
    r.raise_for_status()
    with open(GTF_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    print(f"Downloaded {GTF_PATH}")
else:
    print(f"Reusing cached {GTF_PATH}")

TSS_BED = PROCESSED_DIR / f"gencode.v{latest}.tss.bed"

gene_re = re.compile(r'gene_name "([^"]+)"')
tx_re = re.compile(r'transcript_id "([^"]+)"')

n = 0
with gzip.open(GTF_PATH, "rt") as fin, open(TSS_BED, "w") as fout:
    for line in fin:
        if line.startswith("#"):
            continue
        fields = line.rstrip("\n").split("\t")
        if fields[2] != "transcript":
            continue
        chrom, start, end, strand, attrs = (
            fields[0],
            int(fields[3]),
            int(fields[4]),
            fields[6],
            fields[8],
        )
        gene = gene_re.search(attrs)
        tx = tx_re.search(attrs)
        gene_name = gene.group(1) if gene else "NA"
        tx_id = tx.group(1) if tx else "NA"
        tss = start - 1 if strand == "+" else end - 1  # 0-based TSS
        fout.write(f"{chrom}\t{tss}\t{tss + 1}\t{gene_name}|{tx_id}\t.\t{strand}\n")
        n += 1

print(f"Wrote {n} transcript TSS to {TSS_BED}")
