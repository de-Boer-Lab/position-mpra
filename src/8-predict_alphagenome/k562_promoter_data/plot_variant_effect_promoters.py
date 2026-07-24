"""
Boxplot: does the amount of random DNA inserted near a -150 promoter variant
change its predicted effect on exon2 (HepG2 RNA-seq, AlphaGenome)?

For every gene: exon2 VE = mean(ref_track - alt_track) over the exon2 span
(same offset in every condition, since exon2 never shifts -- see
create_variant_all_promoters.py). One VE value per gene per condition.

x = insertion length (0 = baseline/no-insertion, else 25/50/75/100)
y = exon2 variant effect
color = condition (baseline / upstream -200bp / downstream -100bp insertion)
each point = one gene

Usage
-----
  python plot_variant_effect_promoters.py
  python plot_variant_effect_promoters.py --predictions_dir <path> --out <path.png>
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROMOTER_DIR = "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/8-aphagenome/all_k562_promoters"
DEFAULT_PREDICTIONS_DIR = f"{PROMOTER_DIR}/predictions"
DEFAULT_OUT = f"{PROMOTER_DIR}/exon2_variant_effect_boxplot.png"

CONDITIONS = ["baseline", "upstream", "downstream"]
LENGTH_CATEGORIES = [25, 50, 75, 100]

# Categorical palette (dataviz skill default, slots 1/2/3 -- validated all-pairs CVD-safe)
COLORS = {
    "baseline": "#2a78d6",  # blue
    "upstream": "#eb6834",  # orange
    "downstream": "#1baf7a",  # aqua
}
LABELS = {
    "baseline": "No insertion (variant only)",
    "upstream": "Upstream insertion (-200)",
    "downstream": "Downstream insertion (-100)",
}


def compute_exon2_ve(predictions_dir: str) -> pd.DataFrame:
    meta = pd.read_csv(os.path.join(predictions_dir, "promoters_metadata.tsv"), sep="\t")
    n = len(meta)

    rows = []
    for condition in CONDITIONS:
        ref = np.load(os.path.join(predictions_dir, f"{condition}_ref.npy"), mmap_mode="r")
        alt = np.load(os.path.join(predictions_dir, f"{condition}_alt.npy"), mmap_mode="r")
        assert ref.shape[0] == n, f"{condition}_ref.npy has {ref.shape[0]} rows, metadata has {n}"

        for i, row in meta.iterrows():
            s, e = int(row["exon2_start_offset"]), int(row["exon2_end_offset"])
            ve = float(
                ref[i, s:e].astype(np.float32).mean() - alt[i, s:e].astype(np.float32).mean()
            )
            length = 0 if condition == "baseline" else int(row["length_category"])
            rows.append(
                {
                    "gene": row["gene"],
                    "condition": condition,
                    "length": length,
                    "exon2_ve": ve,
                }
            )

    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    x_positions = {0: 0.0}
    x_positions.update({length: i + 1 for i, length in enumerate(LENGTH_CATEGORIES)})
    dodge = {"baseline": 0.0, "upstream": -0.15, "downstream": 0.15}

    rng = np.random.default_rng(0)
    for condition in CONDITIONS:
        sub = df[df["condition"] == condition]
        lengths = [0] if condition == "baseline" else LENGTH_CATEGORIES
        box_data, positions = [], []
        for length in lengths:
            vals = sub.loc[sub["length"] == length, "exon2_ve"].values
            if len(vals) == 0:
                continue
            box_data.append(vals)
            positions.append(x_positions[length] + dodge[condition])

        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.12,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor=COLORS[condition], edgecolor=COLORS[condition], alpha=0.35),
            medianprops=dict(color=COLORS[condition], linewidth=2),
            whiskerprops=dict(color=COLORS[condition]),
            capprops=dict(color=COLORS[condition]),
        )
        for pos, vals in zip(positions, box_data):
            jitter = rng.uniform(-0.04, 0.04, size=len(vals))
            ax.scatter(
                np.full(len(vals), pos) + jitter,
                vals,
                s=8,
                color=COLORS[condition],
                alpha=0.35,
                linewidths=0,
                zorder=3,
            )

    ax.set_xticks([x_positions[l] for l in [0] + LENGTH_CATEGORIES])
    ax.set_xticklabels(["0\n(no insertion)"] + [str(l) for l in LENGTH_CATEGORIES])
    ax.set_xlabel("Random DNA inserted (bp)")
    ax.set_ylabel("Exon2 variant effect  (mean(ref - alt) over exon2, HepG2 RNA-seq)")
    ax.set_title("-150 promoter variant effect on exon2, by insertion position and length")
    ax.axhline(0, color="#898781", linewidth=1, zorder=0)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=COLORS[c],
            markersize=10,
            label=LABELS[c],
        )
        for c in CONDITIONS
    ]
    ax.legend(handles=handles, loc="best", frameon=False)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exon2 variant-effect boxplot vs. insertion length/position."
    )
    parser.add_argument("--predictions_dir", default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    df = compute_exon2_ve(args.predictions_dir)
    df.to_csv(
        os.path.join(os.path.dirname(args.out), "exon2_variant_effect.tsv"), sep="\t", index=False
    )
    plot(df, args.out)


if __name__ == "__main__":
    main()
