### First read both revcomp and forward var effects
import pandas as pd

forward = pd.read_csv("/scratch/st-cdeboer-1/sambina/position_mpra/outputs/2-opentargets_model_variant_effect/gosai/k562_gosai_1bp.csv.gz", compression="gzip")
reverse = pd.read_csv("/scratch/st-cdeboer-1/sambina/position_mpra/outputs/2-opentargets_model_variant_effect/gosai/k562_gosai_revcomp_1bp.csv.gz", compression="gzip")

# print(forward.head(1))

### Have to switch the column labels for the forward df
rename_dict = {}
for col in forward.columns:
    if col.startswith("offset_"):
        pos = int(col.split("_")[1])
        flipped_col = f"offset_{-pos}"
        rename_dict[col] = flipped_col

# Apply the renaming
forward = forward.rename(columns=rename_dict)
# print(forward.head(1))

### Now change the values
offset_cols = [col for col in forward.columns if col.startswith("offset_")]
forward[offset_cols] = forward[offset_cols].applymap(lambda x: float(x.strip("[]")) if isinstance(x, str) else x)
forward["seq_id"] = ["seq_" + str(i + 1) for i in range(len(forward))]

reverse[offset_cols] = reverse[offset_cols].applymap(lambda x: float(x.strip("[]")) if isinstance(x, str) else x)
reverse["seq_id"] = ["seq_rev_" + str(i + 1) for i in range(len(reverse))]

### Combine and filter out the rows
forward["strand"] = "forward"
reverse["strand"] = "reverse"
combined = pd.concat([forward, reverse], ignore_index=True)
offset_cols = [col for col in combined.columns if col.startswith("offset_")]
# filtered = combined[combined[offset_cols].apply(
#     lambda row: (row > 0.5).any() or (row < -0.5).any(), axis=1)]
filtered = combined

print(len(filtered))

import pandas as pd
import matplotlib.pyplot as plt

# Get the offset columns (those like "offset_90", "offset_-90", etc.)
offset_cols = [col for col in filtered.columns if col.startswith("offset_")]
heatmap_data = filtered[offset_cols]

# Extract numeric positions from column names (e.g. "offset_-90" -> -90)
positions = [int(col.split("_")[1]) for col in offset_cols]

# Count rows with value > 0.5 or < -0.5 at each column
counts = ((heatmap_data > 0.5) | (heatmap_data < -0.5)).sum(axis=0).values

# Convert to percentage of total variants
total_variants = heatmap_data.shape[0]
percentages = counts / total_variants * 100

# Build dataframe for plotting
percent_df = pd.DataFrame({
    "position": positions,
    "percent": percentages
}).sort_values("position")

import matplotlib.pyplot as plt
import seaborn as sns

# Use seaborn style for nicer aesthetics
sns.set(style="whitegrid", context="talk")

plt.figure(figsize=(12,6))

# Line with thicker width and larger markers
plt.plot(
    percent_df["position"], 
    percent_df["percent"], 
    marker="o", 
    markersize=6, 
    linewidth=2.5, 
    color="#4C72B0" 
)

# Set ticks every 20 positions
tick_positions = list(range(-90, 91, 20))
plt.xticks(tick_positions, rotation=45, ha='right')


plt.grid(True, linestyle="--", alpha=0)

plt.tight_layout()

plt.savefig(
    "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/3-cluster/percentage_variants_all.svg",
    format="svg"
)
plt.show()

filtered[offset_cols] = filtered[offset_cols].apply(lambda row: row / row.loc[row.abs().idxmax()], axis=1)
filtered.head(2)

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# new_offset_cols = sorted(renamed_cols.values(), key=lambda x: int(x.split('_')[1]))
new_offset_cols = sorted(offset_cols, key=lambda x: int(x.split('_')[1]))

heatmap_data = filtered[new_offset_cols]
heatmap_data.index = filtered['seq_id']  

colors = ['#3361A5', '#ffffff', '#A52126']

cmap = LinearSegmentedColormap.from_list('custom_red_black_blue', colors, N=256)

### check how many rows here have a sign change 
import pandas as pd

# Boolean mask: row has at least one >0 and at least one <0
mask = (heatmap_data.gt(0).any(axis=1)) & (heatmap_data.lt(0).any(axis=1))

# Count how many rows have both + and -
num_sign_change = mask.sum()

# Total number of rows (variants)
total_rows = len(heatmap_data)

# Percentage
percent_sign_change = (num_sign_change / total_rows) * 100

print(f"Rows with sign change: {num_sign_change}/{total_rows} ({percent_sign_change:.2f}%)")


### Plot the heatmap

g = sns.clustermap(
    heatmap_data,    
    cmap=cmap,
    metric='euclidean', 
    method='complete',
    figsize=(10,20),
    annot=False,           
    col_cluster=False,    
    z_score=None,
    standard_scale=None,
    vmin=-1,
    vmax=1
)

g.cax.set_position([0, .3, .02, .4])
g.cax.set_ylabel("scaled predicted variant effect", rotation=90, labelpad=12)
g.cax.set_visible(False)

positions = list(range(-90, 91, 20))  # -90, -80, ..., 80, 90
tick_positions = [pos + 90 for pos in positions]  # shift positions to 0-based indices

g.ax_heatmap.set_xticks(tick_positions)
g.ax_heatmap.set_xticklabels(positions, rotation=45, ha='right', fontsize=18)
g.ax_heatmap.set_xlabel("Position (wrt centre)")
g.ax_heatmap.set_ylabel("seq_id")
g.fig.suptitle("Clustered heatmap for OpenTarget variants", y=1.001, fontsize=16, ha='center')
plt.savefig(
    "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/cluster/heatmap_OT_all.png",
    dpi=300,               
    bbox_inches="tight"
)
plt.show()

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colorbar import ColorbarBase

colors = ['#3361A5', '#ffffff', '#A52126']
cmap = LinearSegmentedColormap.from_list('custom_red_black_blue', colors, N=256)

fig, ax = plt.subplots(figsize=(2, 6))  # Tall, narrow for colorbar
norm = Normalize(vmin=0, vmax=1)         # Match heatmap scale

cb = ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
cb.set_label("scaled predicted variant effect")

# Remove border and ticks
for spine in ax.spines.values():
    spine.set_visible(False)

ax.tick_params(left=False, right=False, labelleft=True)  # Keep labels, remove ticks
ax.yaxis.set_ticks_position('left')                      # Keep ticks on left side only if you want labels

plt.tight_layout()

# Save the colorbar as SVG and PNG
cb_path_svg = "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/cluster/colorbar.svg"
cb_path_png = "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/cluster/colorbar.png"

fig.savefig(cb_path_svg, dpi=300, bbox_inches='tight', transparent=True)
fig.savefig(cb_path_png, dpi=300, bbox_inches='tight', transparent=True)

plt.show()

