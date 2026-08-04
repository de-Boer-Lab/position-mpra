"""
Pick one canonical transcript per gene from the cached GENCODE GTF and
extract its TSS, preferring MANE_Select (RefSeq/Ensembl matched, protein-
coding only) over Ensembl_canonical (one per gene, all biotypes) over an
arbitrary fallback (lowest transcript_id) for genes with neither tag.
"""

import gzip
import pathlib
import re

RAW_DIR = pathlib.Path(
    "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/8-aphagenome/all_k562_promoters/raw"
)
PROCESSED_DIR = pathlib.Path(
    "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/8-aphagenome/all_k562_promoters/processed"
)

GTF_PATH = next(RAW_DIR.glob("gencode.v*.annotation.gtf.gz"))
OUT_BED = PROCESSED_DIR / "gencode.canonical_gene_tss.bed"

gene_re = re.compile(r'gene_name "([^"]+)"')
tx_re = re.compile(r'transcript_id "([^"]+)"')
tag_re = re.compile(r'tag "([^"]+)"')

best = {}  # gene_name -> (priority, transcript_id, chrom, tss, strand, tag_label)

with gzip.open(GTF_PATH, "rt") as fin:
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
        if not gene or not tx:
            continue
        gene_name, tx_id = gene.group(1), tx.group(1)
        tags = set(tag_re.findall(attrs))

        if "MANE_Select" in tags:
            priority, tag_label = 0, "MANE_Select"
        elif "Ensembl_canonical" in tags:
            priority, tag_label = 1, "Ensembl_canonical"
        else:
            priority, tag_label = 2, "fallback"

        tss = start - 1 if strand == "+" else end - 1  # 0-based

        current = best.get(gene_name)
        if (
            current is None
            or priority < current[0]
            or (priority == current[0] and tx_id < current[1])
        ):
            best[gene_name] = (priority, tx_id, chrom, tss, strand, tag_label)

n_mane = sum(1 for v in best.values() if v[5] == "MANE_Select")
n_canon = sum(1 for v in best.values() if v[5] == "Ensembl_canonical")
n_fallback = sum(1 for v in best.values() if v[5] == "fallback")

with open(OUT_BED, "w") as fout:
    for gene_name, (priority, tx_id, chrom, tss, strand, tag_label) in best.items():
        fout.write(f"{chrom}\t{tss}\t{tss + 1}\t{gene_name}|{tx_id}|{tag_label}\t.\t{strand}\n")

print(
    f"Genes: {len(best)} (MANE_Select: {n_mane}, Ensembl_canonical: {n_canon}, fallback: {n_fallback})"
)
print(f"Wrote {OUT_BED}")
