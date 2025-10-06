"""
Rudimentary 3D DOM Heatmap
==========================

Purpose
- Produce a simple pseudo-3D surface plot (time x price with trade size intensity)
  from metrics generated in `dom_ticks_metrics.csv`.

What it does
- Aggregates trade sizes into a 2D grid by time and price
- Renders a colored surface using Matplotlib's 3D toolkit

Inputs/Outputs
- Input: `dom_ticks_metrics.csv` (produced by `metrics_helper.py`)
- Output: `enhanced_dom_heatmap.png`

Usage
- `python rudimentary_viz.py`

Dependencies
- pandas, numpy, matplotlib
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Load the existing DOM tick data (assuming metrics file has price/time/size columns)
df = pd.read_csv("dom_ticks_metrics.csv")

# Convert timestamps to sequential integer (time axis)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by='timestamp')
df['time_index'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()

# Prepare grid for 3D plotting
price_levels = sorted(df['trade_price'].unique())
time_steps = sorted(df['time_index'].unique())

price_to_idx = {p: i for i, p in enumerate(price_levels)}
time_to_idx = {t: i for i, t in enumerate(time_steps)}

# Create heatmap intensity (e.g. trade size)
heatmap = np.zeros((len(price_levels), len(time_steps)))

for _, row in df.iterrows():
    p_idx = price_to_idx[row['trade_price']]
    t_idx = time_to_idx[row['time_index']]
    heatmap[p_idx, t_idx] += row['trade_size']

# Generate 3D plot
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(time_steps, price_levels)
Z = heatmap

# Normalize for color scale
norm = plt.Normalize(Z.min(), Z.max())
colors = plt.cm.plasma(norm(Z))

# Plot surface
surf = ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1, antialiased=False, linewidth=0)

# Hide the z-axis ticks
ax.set_zticks([])

# Labels
ax.set_xlabel('Time (s)')
ax.set_ylabel('Price')
ax.set_title('3D DOM Heatmap (Trade Sizes)')

# Save to PNG
plt.tight_layout()
output_path = "enhanced_dom_heatmap.png"
plt.savefig(output_path, dpi=300)
output_path
