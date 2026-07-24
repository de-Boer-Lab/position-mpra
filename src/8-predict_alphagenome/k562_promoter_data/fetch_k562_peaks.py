"""
Download K562 ATAC-seq IDR thresholded peaks (GRCh38) from ENCODE.

Experiment ENCSR483RKN was chosen among the 10 K562 ATAC-seq experiments
on ENCODE (GRCh38) because it is processed with the standard ENCODE
pipeline (Michael Snyder lab) and has zero ERROR-level audits, unlike the
newer Greenleaf-lab experiments which carry ERROR audits.
"""

import pathlib
import requests

OUT_DIR = pathlib.Path(
    "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/8-aphagenome/all_k562_promoters/raw"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILE_ACCESSION = "ENCFF117MSK"
URL = f"https://www.encodeproject.org/files/{FILE_ACCESSION}/@@download/{FILE_ACCESSION}.bed.gz"
DEST = OUT_DIR / "k562_atac_idr_peaks.GRCh38.bed.gz"

r = requests.get(URL, stream=True)
r.raise_for_status()
with open(DEST, "wb") as f:
    for chunk in r.iter_content(chunk_size=1 << 20):
        f.write(chunk)

print(f"Downloaded {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
