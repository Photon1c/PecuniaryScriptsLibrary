"""
DOM Heatmap v2 with Trade Flow Bubbles
======================================

Purpose
- Enhanced heatmap that overlays directional trade bubbles on top of the
  price/time liquidity field.

What it does
- Bins time and price for a turbo-colored heatmap
- Adds bubble markers sized by `trade_size` and colored by trade direction

Inputs/Outputs
- Input: `dom_ticks_metrics.csv`
- Output: `dom_heatmap_with_bubbles.png`

Usage
- `python heatmap_v2.py`

Dependencies
- pandas, matplotlib, numpy
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# Load the metrics-enhanced DOM tick data
df = pd.read_csv("dom_ticks_metrics.csv")

# Convert timestamp to datetime and normalize to seconds since start
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["time_offset"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds()

# Round prices for heatmap binning
df["price_bin"] = df["trade_price"].round(2)

# Pivot table to form heatmap structure
heatmap_data = df.pivot_table(index="price_bin", columns="time_offset", values="trade_size", aggfunc="sum", fill_value=0)

# Normalize data for heatmap color scaling
norm = mcolors.Normalize(vmin=heatmap_data.values.min(), vmax=heatmap_data.values.max())
cmap = cm.get_cmap("turbo")

# Create the base heatmap
fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(heatmap_data, aspect="auto", cmap=cmap, norm=norm, origin="lower",
               extent=[heatmap_data.columns.min(), heatmap_data.columns.max(),
                       heatmap_data.index.min(), heatmap_data.index.max()])

# Add 3D-style trade bubbles
for _, row in df.iterrows():
    x = row["time_offset"]
    y = row["trade_price"]
    size = row["trade_size"]
    color = "green" if row["trade_direction"] > 0 else "red"
    alpha = 0.4 + min(size / df["trade_size"].max(), 1) * 0.5
    ax.scatter(x, y, s=20 + size, c=color, alpha=alpha, edgecolors='black', linewidths=0.2)

# Add colorbar and labels
cbar = fig.colorbar(im, ax=ax, label="Trade Size Heat")
ax.set_xlabel("Seconds Since Start")
ax.set_ylabel("Price")
ax.set_title("Enhanced DOM Heatmap with Trade Flow Bubbles")

# Save output
output_path = "dom_heatmap_with_bubbles.png"
plt.savefig(output_path, dpi=300)
plt.close()

output_path
