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

Also produces two correlation heatmaps (Pearson R^2 between the no-insertion
exon2 VE and the exon2 VE at each insertion length, one for downstream, one
for upstream) and a 3x3 scatterplot grid, one panel per condition in
CONDITION_KEYS (baseline + 4 upstream lengths + 4 downstream lengths), x =
baseline exon2 VE, y = that condition's exon2 VE.

Reads the 9-condition-per-gene layout written by the current
create_variant_all_promoters.py ({condition_key}_ref.npy/_alt.npy for
condition_key in baseline/upstream_{25,50,75,100}/downstream_{25,50,75,100}).

Usage
-----
  python plot_variant_effect_promoters.py
  python plot_variant_effect_promoters.py --predictions_dir <path> --out <path.png>
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from tqdm import tqdm

PROMOTER_DIR = "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/8-aphagenome/all_k562_promoters"
DEFAULT_PREDICTIONS_DIR = f"{PROMOTER_DIR}/predictions"
DEFAULT_OUT = f"{PROMOTER_DIR}/exon2_variant_effect_boxplot.png"
DEFAULT_HEATMAP_OUT = f"{PROMOTER_DIR}/exon2_variant_effect_heatmap.png"
DEFAULT_CORR_OUT = f"{PROMOTER_DIR}/exon2_ve_correlation_heatmap.png"
DEFAULT_CORR_UPSTREAM_OUT = f"{PROMOTER_DIR}/exon2_ve_correlation_heatmap_upstream.png"
DEFAULT_SCATTER_OUT = f"{PROMOTER_DIR}/exon2_ve_scatter_grid.png"

CONDITIONS = ["baseline", "upstream", "downstream"]
LENGTH_CATEGORIES = [25, 50, 75, 100]

# Matches CONDITION_KEYS in create_variant_all_promoters.py -- the 9 files
# ({condition_key}_ref.npy/_alt.npy) actually written to predictions_dir.
CONDITION_KEYS = ["baseline"] + [
    f"{position}_{k}" for position in ("upstream", "downstream") for k in LENGTH_CATEGORIES
]

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

# Column order for the heatmap: no random sequence, then -100 (downstream), then -200 (upstream).
HEATMAP_CONDITIONS = ["baseline", "downstream", "upstream"]
HEATMAP_COLUMN_LABELS = ["None", "-100", "-200"]

# Diverging pair (dataviz skill default): blue <-> red, gray midpoint at 0.
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"], N=256
)

# Sequential white->blue ramp for a 0-1 magnitude (R^2), matching plot_agarwal.ipynb.
SEQUENTIAL_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "sequential_blue", ["#ffffff", "#3361A5"], N=256
)


def _compute_condition_chunk(
    predictions_dir: str,
    condition_key: str,
    condition: str,
    fixed_length,
    gene_ids_chunk,
    starts_chunk,
    ends_chunk,
    lengths_chunk,
    row_start: int,
) -> list:
    """Runs in a worker process: computes VE for meta rows
    [row_start, row_start + len(gene_ids_chunk)) of one condition_key. Each
    worker reopens the memmap itself -- read-only mmaps are safely shared
    across processes, nothing is duplicated in memory up front."""
    ref = np.load(os.path.join(predictions_dir, f"{condition_key}_ref.npy"), mmap_mode="r")
    alt = np.load(os.path.join(predictions_dir, f"{condition_key}_alt.npy"), mmap_mode="r")

    out = []
    for local_idx, gene in enumerate(gene_ids_chunk):
        i = row_start + local_idx
        s, e = int(starts_chunk[local_idx]), int(ends_chunk[local_idx])
        ve = float(ref[i, s:e].astype(np.float32).mean() - alt[i, s:e].astype(np.float32).mean())
        length = fixed_length if fixed_length is not None else int(lengths_chunk[local_idx])
        out.append((gene, condition, length, ve))
    return out


def compute_exon2_ve(predictions_dir: str, workers: int = None) -> pd.DataFrame:
    """Reads whichever schema is actually in predictions_dir:
      - old (one length/gene): promoters_metadata.tsv has "length_category";
        files are {baseline,upstream,downstream}_{ref,alt}.npy.
      - new (all 4 lengths x both positions, every gene): metadata has
        insertion_seq_{25,50,75,100} instead; files are one per CONDITION_KEY
        (baseline, upstream_{K}, downstream_{K}).
    Either way, returns the same tidy columns: gene, condition
    (baseline/upstream/downstream), length (0/25/50/75/100), exon2_ve.

    Parallelized across processes: each condition is split into chunks of
    contiguous gene rows, and all (condition, chunk) tasks across all
    conditions run concurrently in a process pool -- this is what actually
    buys the speedup, since the per-row cost here is dominated by mmap I/O
    latency on network storage, and concurrent readers hide that latency
    instead of paying it one row at a time.
    """
    meta = pd.read_csv(os.path.join(predictions_dir, "promoters_metadata.tsv"), sep="\t")
    n = len(meta)
    old_schema = "length_category" in meta.columns
    condition_keys = CONDITIONS if old_schema else CONDITION_KEYS

    gene_ids = meta["gene"].to_numpy()
    starts = meta["exon2_start_offset"].to_numpy()
    ends = meta["exon2_end_offset"].to_numpy()
    length_categories = meta["length_category"].to_numpy() if old_schema else None

    workers = workers or len(os.sched_getaffinity(0))
    chunk_size = max(1, -(-n // workers))  # ~1 chunk per worker per condition

    tasks = []
    for condition_key in condition_keys:
        if condition_key == "baseline":
            condition, fixed_length = "baseline", 0
        elif old_schema:
            condition, fixed_length = condition_key, None  # length varies per gene
        else:
            condition, length_str = condition_key.rsplit("_", 1)
            fixed_length = int(length_str)

        for row_start in range(0, n, chunk_size):
            row_end = min(row_start + chunk_size, n)
            tasks.append(
                (
                    predictions_dir,
                    condition_key,
                    condition,
                    fixed_length,
                    gene_ids[row_start:row_end],
                    starts[row_start:row_end],
                    ends[row_start:row_end],
                    length_categories[row_start:row_end] if length_categories is not None else None,
                    row_start,
                )
            )

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_compute_condition_chunk, *task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="chunks"):
            rows.extend(future.result())

    return pd.DataFrame(rows, columns=["gene", "condition", "length", "exon2_ve"])


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


def plot_heatmap(df: pd.DataFrame, out_path: str) -> None:
    """Small multiples: one heatmap per insertion length, rows = genes with that
    length category, columns = no random sequence / random at -100 / random at
    -200. Each gene has exactly one length_category, so the four panels
    partition all genes (they do not repeat a gene at multiple lengths)."""
    baseline_ve = df.loc[df["condition"] == "baseline"].set_index("gene")["exon2_ve"]

    panels = []
    for length in LENGTH_CATEGORIES:
        genes = sorted(df.loc[df["length"] == length, "gene"].unique())
        sub = df[(df["length"] == length) | (df["condition"] == "baseline")]
        sub = sub[sub["gene"].isin(genes)]
        pivot = sub.pivot_table(index="gene", columns="condition", values="exon2_ve")
        pivot = pivot.reindex(columns=HEATMAP_CONDITIONS)
        pivot = pivot.loc[baseline_ve.reindex(pivot.index).sort_values(ascending=False).index]
        panels.append((length, pivot))

    vmax = max(np.nanpercentile(np.abs(p.values), 99) for _, p in panels)
    vmax = max(vmax, 1e-12)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 8), sharey=False)
    im = None
    for ax, (length, pivot) in zip(axes, panels, strict=True):
        im = ax.imshow(
            pivot.values,
            aspect="auto",
            cmap=DIVERGING_CMAP,
            norm=norm,
            interpolation="nearest",
        )
        ax.set_xticks(range(len(HEATMAP_CONDITIONS)))
        ax.set_xticklabels(HEATMAP_COLUMN_LABELS, fontsize=8)
        ax.set_title(f"{length} bp insertion\n(n={len(pivot)} genes)", fontsize=10)
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("Genes (sorted by no-insertion effect)")

    fig.suptitle("Exon2 variant effect by insertion length and position", y=1.02)
    fig.text(
        0.5,
        -0.02,
        "Columns: no random sequence / random sequence inserted at -100 / at -200 (bp relative to TSS)",
        ha="center",
        fontsize=9,
        color="#52514e",
    )
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("Exon2 variant effect  (mean(ref - alt))")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


def plot_correlation_heatmap(
    df: pd.DataFrame,
    out_path: str,
    condition: str,
    labels: list,
    axis_label: str,
    condition_label: str,
) -> None:
    """Strictly-lower-triangle heatmap of all-by-all Pearson R^2 between exon2
    VE at each insertion length for `condition`, plus the no-insertion
    baseline (length 0). The upper triangle and the diagonal (trivial R^2=1
    self-correlation) are both masked out."""
    lengths_order = [100, 75, 50, 25, 0]
    n = len(lengths_order)

    series = {}
    for length in lengths_order:
        cond = "baseline" if length == 0 else condition
        series[length] = df.loc[
            (df["condition"] == cond) & (df["length"] == length)
        ].set_index("gene")["exon2_ve"]

    r2_matrix = np.full((n, n), np.nan)
    for i, li in enumerate(lengths_order):
        for j, lj in enumerate(lengths_order):
            if j >= i:
                continue
            genes = series[li].index.intersection(series[lj].index)
            x = series[li].reindex(genes).values
            y = series[lj].reindex(genes).values
            r2_matrix[i, j] = np.corrcoef(x, y)[0, 1] ** 2

    masked = np.ma.masked_invalid(r2_matrix)
    cmap = SEQUENTIAL_BLUE_CMAP.copy()
    cmap.set_bad(color="none")

    fig, ax = plt.subplots(figsize=(1.2 * n + 1, 1.2 * n + 1))
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.set_xlabel(axis_label)
    ax.set_ylabel(axis_label)
    ax.set_title(f"Pairwise correlation of exon2 variant effect\n(Pearson R$^2$, {condition_label})")

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.03)
    cbar.set_label("Pearson R$^2$")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


def plot_scatter_grid(df: pd.DataFrame, out_path: str) -> None:
    """3x3 grid, one scatterplot per condition in CONDITION_KEYS: x = baseline
    (no-insertion) exon2 VE, y = that condition's exon2 VE, one point per
    gene. The baseline-vs-baseline panel is a trivial sanity check."""
    baseline_ve = df.loc[df["condition"] == "baseline"].set_index("gene")["exon2_ve"]

    panels = [("baseline", 0)] + [
        (position, k) for position in ("upstream", "downstream") for k in LENGTH_CATEGORIES
    ]

    fig, axes = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey=True)
    for ax, (position, length) in zip(axes.flat, panels):
        sub = df[(df["condition"] == position) & (df["length"] == length)]
        x = baseline_ve.reindex(sub["gene"]).values
        y = sub["exon2_ve"].values
        r2 = np.corrcoef(x, y)[0, 1] ** 2

        ax.scatter(x, y, s=6, color=COLORS[position], alpha=0.3, linewidths=0)

        title = "No insertion (self)" if position == "baseline" else f"{LABELS[position]}, {length}bp"
        ax.set_title(f"{title}\nR$^2$={r2:.2f}", fontsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("Baseline exon2 VE")
    for ax in axes[:, 0]:
        ax.set_ylabel("Condition exon2 VE")

    fig.suptitle("Exon2 variant effect: no-insertion baseline vs. each insertion condition", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exon2 variant-effect boxplot vs. insertion length/position."
    )
    parser.add_argument("--predictions_dir", default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--heatmap_out", default=DEFAULT_HEATMAP_OUT)
    parser.add_argument("--corr_out", default=DEFAULT_CORR_OUT)
    parser.add_argument("--corr_upstream_out", default=DEFAULT_CORR_UPSTREAM_OUT)
    parser.add_argument("--scatter_out", default=DEFAULT_SCATTER_OUT)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Process pool size for compute_exon2_ve (default: CPUs allocated to this job).",
    )
    args = parser.parse_args()

    ve_path = os.path.join(os.path.dirname(args.out), "exon2_variant_effect.tsv")
    if os.path.exists(ve_path):
        print(f"Found existing {ve_path}, skipping VE computation")
        df = pd.read_csv(ve_path, sep="\t")
    else:
        df = compute_exon2_ve(args.predictions_dir, workers=args.workers)
        df.to_csv(ve_path, sep="\t", index=False)

    plot(df, args.out)
    plot_heatmap(df, args.heatmap_out)

    lengths_order = [100, 75, 50, 25, 0]
    plot_correlation_heatmap(
        df,
        args.corr_out,
        condition="downstream",
        labels=[str(-(150 + length)) for length in lengths_order],
        axis_label="Variant position relative to TSS (bp)  [= -150 - insertion length]",
        condition_label="downstream insertion",
    )
    plot_correlation_heatmap(
        df,
        args.corr_upstream_out,
        condition="upstream",
        labels=[str(length) for length in lengths_order],
        axis_label="Random sequence added upstream of the variant (bp)",
        condition_label="upstream insertion",
    )
    plot_scatter_grid(df, args.scatter_out)


if __name__ == "__main__":
    main()
