"""
DOM Heatmap v1 (Seaborn)
========================

Purpose
- Build a combined (bid + ask) liquidity heatmap over time and price.

What it does
- Rounds timestamps to 1-second buckets
- Builds separate bid/ask grids and sums them for a composite heatmap

Inputs/Outputs
- Input: `dom_ticks_metrics.csv`
- Output: `dom_heatmap.png`

Usage
- `python heatmap_v1.py`

Dependencies
- pandas, numpy, matplotlib, seaborn
"""
# Re-import needed modules after kernel reset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib

# Set non-interactive backend
matplotlib.use('Agg')

# Reload the CSV file
csv_path = "dom_ticks_metrics.csv"
df = pd.read_csv(csv_path)

# Convert timestamps to datetime and round to nearest second
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["timestamp_rounded"] = df["timestamp"].dt.round("1s")

# Create sorted lists of price levels and timestamps
# Rebuild time_stamps and price_levels just in case
price_levels = sorted(set(df["bid_price"]).union(df["ask_price"]))
time_stamps = sorted(set(df["timestamp_rounded"]))

# Rebuild index maps
price_index = {price: idx for idx, price in enumerate(price_levels)}
time_index = {time: idx for idx, time in enumerate(time_stamps)}

# Re-initialize grids
bid_grid = np.zeros((len(price_levels), len(time_stamps)))
ask_grid = np.zeros((len(price_levels), len(time_stamps)))

# Safely populate grids
for _, row in df.iterrows():
    try:
        t_idx = time_index[row["timestamp_rounded"]]
        bid_idx = price_index[row["bid_price"]]
        ask_idx = price_index[row["ask_price"]]
        bid_grid[bid_idx, t_idx] += row["bid_size"]
        ask_grid[ask_idx, t_idx] += row["ask_size"]
    except KeyError:
        continue  # Skip any row with mismatched keys

# Create combined heatmap
fig, ax = plt.subplots(figsize=(12, 8))
combined_grid = bid_grid + ask_grid
norm = mcolors.PowerNorm(gamma=0.5, vmin=np.min(combined_grid), vmax=np.max(combined_grid))

sns.heatmap(
    combined_grid,
    ax=ax,
    cmap="turbo",
    norm=norm,
    cbar_kws={'label': 'Order Size'},
    xticklabels=[t.strftime("%H:%M:%S") for t in time_stamps],
    yticklabels=[f"{p:.2f}" for p in price_levels]
)

ax.set_title("DOM Heatmap (Bid + Ask Sizes)")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
plt.xticks(rotation=45)
plt.tight_layout()

# Save to PNG
output_path = "dom_heatmap.png"
plt.savefig(output_path)
output_path
