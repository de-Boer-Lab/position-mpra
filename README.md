# Position dependent variant effects reveal importance of context in genomic regulation
This repository contains the analysis code and supporting materials for studying how the positional context of genetic variants influences regulatory activity. The project integrates experimental variant effect data, model-based predictions, clustering, and in silico mutagenesis to characterize context-dependent regulatory mechanisms.

## Directory Structure

```text
├── README.md
├── environment.yml
├── figures
└── src
    ├── 1-GTeX_experimental_variant_effect
    ├── 2-opentargets_model_variant_effect
    ├── 3-cluster_variant_effects
    ├── 4-range_variant_effect_TFs
    ├── 5-TF_position
    └── 6-ism_variant_effects
```

## Environment and Setup

All analyses are designed to be reproducible using Conda.

`conda env create -f environment.yml
conda activate variant-context`


## Citation

If you use this code, please cite:

> Position-dependent variant effects reveal the importance of context in genomic regulation.

(Preprint / manuscript details to be added.)