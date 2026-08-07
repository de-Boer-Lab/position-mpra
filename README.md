# Position dependent variant effects reveal importance of context in genomic regulation
This repository contains the analysis code and supporting materials for studying how the positional context of genetic variants influences regulatory activity. The project integrates experimental variant effect data, model-based predictions, clustering, and in silico mutagenesis to characterize context-dependent regulatory mechanisms.

## Directory Structure

```text
├── README.md
├── figures
└── src
    ├── 1-GTeX_experimental_variant_effect
    ├── 2-opentargets_model_variant_effect
    ├── 3-cluster_variant_effects
    ├── 4-range_variant_effect_TFs
    ├── 5-TF_position
    ├── 6-ism_variant_effects
    ├── 7-investigating_sines
    ├── 8-predict_alphagenome
    └── 9-schematic
```

## Environment and Setup

This repo does not ship a single consolidated `environment.yml` -- different
scripts were developed and run under different Conda environments (e.g.
`dream_rocky_3` for general analysis/plotting, `alphagenome_pt` for the
AlphaGenome PyTorch prediction scripts under `8-predict_alphagenome`, plus a
few other model-specific environments). To run a given script, use whichever
environment it was originally run under -- most scripts and SLURM wrapper
(`run_*.sh`) files note this via a `conda activate <env_name>` line near the
top; match that when setting up your own environment.


## Citation

If you use this code, please cite:

> Position-dependent variant effects reveal the importance of context in genomic regulation.
> bioRxiv. https://doi.org/10.64898/2026.03.17.712488