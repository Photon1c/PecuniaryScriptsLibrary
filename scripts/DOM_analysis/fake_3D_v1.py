"""
Fake 3D Heatmap v1
==================

Purpose
- First iteration of a 2D heatmap with translucent circles to mimic 3D depth.

What it does
- Converts timestamps to seconds since start
- Pivots trade sizes into a time x price grid
- Overlays circular markers scaled by trade size

Inputs/Outputs
- Input: `dom_ticks_metrics.csv`
- Output: `fake_3d_dom_heatmap.png`

Usage
- `python fake_3D_v1.py`

Dependencies
- pandas, matplotlib, numpy
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import ListedColormap

# Load the DOM tick metrics data
df = pd.read_csv("dom_ticks_metrics.csv")

# Convert timestamp to seconds since start
df['timestamp'] = pd.to_datetime(df['timestamp'])
start_time = df['timestamp'].iloc[0]
df['seconds'] = (df['timestamp'] - start_time).dt.total_seconds()

# Pivot the data to get a heatmap (price vs time)
heatmap_data = df.pivot_table(index='trade_price', columns='seconds', values='trade_size', aggfunc='sum', fill_value=0)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the heatmap
cmap = plt.get_cmap('plasma')
cax = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', origin='lower', extent=[
    df['seconds'].min(), df['seconds'].max(),
    df['trade_price'].min(), df['trade_price'].max()
])
fig.colorbar(cax, ax=ax, label='Trade Size Heat')

# Overlay fake 3D bubbles (circles with radial gradient-like alpha simulation)
for _, row in df.iterrows():
    alpha = min(0.7, 0.2 + row['trade_size'] / df['trade_size'].max())
    radius = 0.2 + 0.3 * (row['trade_size'] / df['trade_size'].max())
    circle = patches.Circle((row['seconds'], row['trade_price']), radius, color='limegreen', alpha=alpha, zorder=3)
    ax.add_patch(circle)

# Labels and layout
ax.set_title("2D DOM Heatmap with 3D-Illusion Trade Bubbles")
ax.set_xlabel("Seconds Since Start")
ax.set_ylabel("Price")

# Save the result
output_path = "fake_3d_dom_heatmap.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(output_path)
