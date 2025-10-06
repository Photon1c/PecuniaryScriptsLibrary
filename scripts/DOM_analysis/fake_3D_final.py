"""
Turbo Heatmap with Pseudo‑3D Trade Bubbles (Final)
=================================================

Purpose
- Visualize DOM snapshots with a smoothed heatmap and overlaid bubble glyphs
  to suggest depth and activity intensity.

What it does
- Loads `dom_snapshot_log.csv`, cleans numeric fields, computes seconds offset
- Builds a 2D histogram (time x price) weighted by bid/ask sizes
- Applies Gaussian smoothing and overlays circle bubbles representing trades

Inputs/Outputs
- Input: `dom_snapshot_log.csv`
- Output: `final_dom_heatmap.png`

Usage
- `python fake_3D_final.py`

Dependencies
- pandas, numpy, matplotlib, scipy

Notes
- Columns are expected to exist; adjust selectors/column names if your CSV differs.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import random

# Replace this line:
# color = 'green' if row['spoofing_event'] else 'red'

# With a fallback:
color = random.choice(['green', 'red'])  # Or just 'red' if you want consistency


# Load the sample DOM snapshot
df = pd.read_csv("dom_snapshot_log.csv")
# Remove rows that contain headers or non-numeric entries
df = df[df["bid_price"].str.replace('.', '', 1).str.isnumeric()]

# Convert numeric fields (strip commas and convert to float)
numeric_cols = ['bid_price', 'bid_size', 'ask_price', 'ask_size', 'last_trade_price', 'last_trade_size']
for col in numeric_cols:
    df[col] = df[col].astype(str).str.replace(',', '').astype(float)
if 'spoofing_event' not in df.columns:
    df['spoofing_event'] = False  # Default to False


print(df.head(5))
print(df.columns)

# Preprocessing
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", infer_datetime_format=True)

df["seconds"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds()


# Remove commas and convert bid/ask sizes to numeric
df["bid_size"] = pd.to_numeric(df["bid_size"].astype(str).str.replace(",", ""), errors="coerce")
df["ask_size"] = pd.to_numeric(df["ask_size"].astype(str).str.replace(",", ""), errors="coerce")


#Clean up

required_columns = ["timestamp", "last_trade_price", "bid_size", "ask_size"]
df = df.dropna(subset=[col for col in required_columns if col in df.columns])
df["seconds"] = pd.to_numeric(df["seconds"], errors="coerce")
df["last_trade_price"] = pd.to_numeric(df["last_trade_price"], errors="coerce")
df = df.dropna(subset=["seconds", "last_trade_price"])


# Create a 2D histogram heatmap (price vs time)
heatmap_bins = (300, 300)
heatmap, xedges, yedges = np.histogram2d(df["seconds"], df["last_trade_price"],
                                         bins=heatmap_bins,
                                         weights=df["ask_size"] + df["bid_size"])

# Apply Gaussian smoothing for smoother heatmap
heatmap_smooth = gaussian_filter(heatmap, sigma=2)

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

# Plot heatmap
cmap = cm.get_cmap('turbo')
im = ax.imshow(heatmap_smooth.T, extent=extent, origin='lower', aspect='auto', cmap=cmap, norm=Normalize())

# Overlay pseudo-3D trade bubbles
for _, row in df.iterrows():
    bubble_size = np.sqrt(row['last_trade_size'] + 1) * 3

    color = 'green' if row['spoofing_event'] else 'red'
    bubble = Circle((row["seconds"], row["last_trade_price"]),
                    radius=bubble_size,
                    color=color,
                    alpha=0.25,
                    ec='black',
                    linewidth=1)
    ax.add_patch(bubble)

# Right-side DOM bar chart
dom_summary = df.groupby("last_trade_price")[["bid_size", "ask_size"]].sum()
prices = dom_summary.index
bid_sizes = dom_summary["bid_size"].values
ask_sizes = dom_summary["ask_size"].values

ax2 = ax.twinx()
ax2.barh(prices, bid_sizes, color='green', alpha=0.3, height=0.1, label='Bid')
ax2.barh(prices, -ask_sizes, color='red', alpha=0.3, height=0.1, label='Ask')
ax2.set_ylim(ax.get_ylim())
ax2.axis('off')

# Style
ax.set_title("Turbo Heatmap with Pseudo-3D Trade Bubbles and DOM Profile", fontsize=14)
ax.set_xlabel("Seconds Since Start")
ax.set_ylabel("Price")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Liquidity Heat")

# Grid styling
ax.grid(color='white', linestyle='--', alpha=0.3)

# Save the figure
output_path = "final_dom_heatmap.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300)
output_path
