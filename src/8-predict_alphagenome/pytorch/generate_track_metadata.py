"""Generate track_metadata_human.parquet for alphagenome_pytorch.

Run once with:
    conda run -n alphagenome_env python generate_track_metadata.py

Requires:
    - alphagenome (JAX SDK, in alphagenome_env)
    - pandas, pyarrow
    - ALPHAGENOME_API_KEY set in ../.env
"""

import os
import sys
from pathlib import Path

# Remove the script's own directory from sys.path to avoid shadowing stdlib/site-packages
# with local files (e.g. testing.py shadows anndata's `testing` package)
_script_dir = str(Path(__file__).parent.resolve())
sys.path = [p for p in sys.path if Path(p).resolve() != Path(_script_dir)]

# Load API key from .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

api_key = os.environ.get("ALPHAGENOME_API_KEY")
if not api_key:
    sys.exit("ALPHAGENOME_API_KEY not found. Check ../.env")

import alphagenome.models.dna_client as client
from alphagenome.models.dna_model import Organism
import pandas as pd

# alphagenome_pytorch data directory (fixed path, not importable from this env)
DATA_DIR = Path(
    "/arc/project/st-cdeboer-1/sambina/miniconda3/envs/alphagenome_pt"
    "/lib/python3.12/site-packages/alphagenome_pytorch/data"
)
out_human = DATA_DIR / "track_metadata_human.parquet"

print(f"Connecting to AlphaGenome API...")
model = client.create(api_key=api_key, timeout=60)

print("Fetching human track metadata...")
meta = model.output_metadata(organism=Organism.HOMO_SAPIENS)

rows = []
# meta.rna_seq etc. are pandas DataFrames (TrackMetadata = pd.DataFrame in JAX)
output_map = {
    "atac": meta.atac,
    "dnase": meta.dnase,
    "procap": meta.procap,
    "cage": meta.cage,
    "rna_seq": meta.rna_seq,
    "chip_tf": meta.chip_tf,
    "chip_histone": meta.chip_histone,
}

for output_name, df_raw in output_map.items():
    if df_raw is None:
        continue
    df = df_raw.copy().reset_index(drop=True)
    df.insert(0, "output_type", output_name)
    df.insert(1, "track_index", range(len(df)))
    df.insert(2, "organism", 0)
    # Rename JAX 'name' column to 'track_name' for alphagenome_pytorch
    if "name" in df.columns and "track_name" not in df.columns:
        df = df.rename(columns={"name": "track_name"})
    rows.append(df)

combined = pd.concat(rows, ignore_index=True)
combined.to_parquet(out_human, index=False)
print(f"Saved {len(combined)} rows to:\n  {out_human}")
print(f"\nRNA-seq tracks: {combined[combined.output_type == 'rna_seq'].shape[0]}")

# Quick sanity check for HepG2
rna = combined[combined.output_type == "rna_seq"]
if "ontology_curie" in rna.columns:
    hepg2 = rna[rna.ontology_curie == "EFO:0001187"]
    if hepg2.empty and "track_name" in rna.columns:
        hepg2 = rna[rna.track_name.str.contains("HepG2", case=False, na=False)]
    if not hepg2.empty:
        print(f"\nHepG2 RNA-seq tracks found:")
        for _, row in hepg2.iterrows():
            print(f"  [{int(row.track_index)}] {row.get('track_name', '?')}")
