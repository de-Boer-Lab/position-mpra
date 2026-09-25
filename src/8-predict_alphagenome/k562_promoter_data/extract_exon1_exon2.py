"""
For the canonical transcripts underlying k562_tss_in_open_chromatin.canonical_gene.bed,
extract exon 1 and exon 2 coordinates (GENCODE's exon_number is already
transcript-relative / strand-aware, so exon_number 1 is always the most 5' exon).
"""

import gzip
import pathlib
import re

RAW_DIR = pathlib.Path(
    "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/8-aphagenome/all_k562_promoters/raw"
)
OUT_DIR = pathlib.Path(
    "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/8-aphagenome/all_k562_promoters"
)

TSS_BED = OUT_DIR / "k562_tss_in_open_chromatin.canonical_gene.bed"
GTF_PATH = next(RAW_DIR.glob("gencode.v*.annotation.gtf.gz"))
OUT_BED = OUT_DIR / "k562_exon1_exon2.canonical_gene.bed"

# transcript_id (versioned) -> gene_name, from the col4 = "gene|transcript_id|tag" field
tx_to_gene = {}
with open(TSS_BED) as f:
    for line in f:
        gene, tx_id, tag = line.rstrip("\n").split("\t")[3].split("|")
        tx_to_gene[tx_id] = gene

tx_re = re.compile(r'transcript_id "([^"]+)"')
exon_re = re.compile(r"exon_number (\d+)")

rows = []
with gzip.open(GTF_PATH, "rt") as fin:
    for line in fin:
        if line.startswith("#"):
            continue
        fields = line.rstrip("\n").split("\t")
        if fields[2] != "exon":
            continue
        attrs = fields[8]
        tx = tx_re.search(attrs)
        if not tx or tx.group(1) not in tx_to_gene:
            continue
        exon_num = exon_re.search(attrs)
        if not exon_num or exon_num.group(1) not in ("1", "2"):
            continue
        chrom, start, end, strand = fields[0], int(fields[3]), int(fields[4]), fields[6]
        gene = tx_to_gene[tx.group(1)]
        rows.append(
            (chrom, start - 1, end, f"{gene}|{tx.group(1)}|exon{exon_num.group(1)}", ".", strand)
        )

rows.sort(key=lambda r: (r[0], r[1]))
with open(OUT_BED, "w") as fout:
    for r in rows:
        fout.write("\t".join(map(str, r)) + "\n")

n_genes = len(tx_to_gene)
n_exon1 = sum(1 for r in rows if r[3].endswith("exon1"))
n_exon2 = sum(1 for r in rows if r[3].endswith("exon2"))
print(f"Genes queried: {n_genes}")
print(
    f"Exon1 rows: {n_exon1}  Exon2 rows: {n_exon2}  (genes missing exon2 are single-exon transcripts)"
)
print(f"Wrote {OUT_BED}")
