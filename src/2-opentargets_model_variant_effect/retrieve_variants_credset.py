"""Retrieve fine-mapped GWAS SNVs from Open Targets Genetics 22.09 credible sets.

Recreates data/opentargets_model_data/snvs_only_gwas_0.1_postprob.csv:
unique GWAS tag variants from the v2d_credset table with postprob >= 0.1,
restricted to SNVs (single-base ref and alt).

Usage: python retrieve_variants_credset.py [output_csv]
"""
import io
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

OT_BASE = "https://ftp.ebi.ac.uk/pub/databases/opentargets/genetics/22.09"
OUTPUT = (
    sys.argv[1] if len(sys.argv) > 1
    else "/scratch/st-cdeboer-1/sambina/position_mpra/data/opentargets_model_data/snvs_only_gwas_0.1_postprob.csv"
)
POSTPROB_THRESHOLD = 0.1
MAX_WORKERS = 6
COLS = ["type", "tag_variant_id", "tag_chrom", "tag_pos", "tag_ref", "tag_alt", "postprob"]


def fetch(url):
    with urllib.request.urlopen(url, timeout=600) as resp:
        return resp.read()


def fetch_and_filter(url):
    df = pd.read_parquet(io.BytesIO(fetch(url)), columns=COLS)
    snv = (df["tag_ref"].str.len() == 1) & (df["tag_alt"].str.len() == 1)
    df = df[(df["type"] == "gwas") & (df["postprob"] >= POSTPROB_THRESHOLD) & snv]
    return df.drop_duplicates("tag_variant_id")


table_url = f"{OT_BASE}/v2d_credset/"
parts = sorted(set(re.findall(r'href="(part-[^"]+\.parquet)"', fetch(table_url).decode())))
print(f"Found {len(parts)} v2d_credset parquet files")

chunks = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    for i, df in enumerate(pool.map(fetch_and_filter, [table_url + p for p in parts]), 1):
        chunks.append(df)
        print(f"[{i:3d}/{len(parts)}] {len(df):,} rows kept", flush=True)

result = (
    pd.concat(chunks, ignore_index=True)
    .drop_duplicates("tag_variant_id")
    .rename(columns={"tag_chrom": "CHR_ID", "tag_pos": "Position", "tag_ref": "Ref", "tag_alt": "Alt"})
    [["tag_variant_id", "CHR_ID", "Position", "Ref", "Alt"]]
    .sort_values(["CHR_ID", "Position", "Ref", "Alt"])
)
result.to_csv(OUTPUT, index=False)
print(f"Saved {len(result):,} variants -> {OUTPUT}")
