"""
Metrics Helper for DOM Ticks
============================

Purpose
- Compute baseline metrics from raw DOM tick streams to enable downstream
  visualization and analytics.

What it does
- Cleans numeric fields, drops incomplete rows
- Computes spread, mid price, spread percent, size imbalance, trade direction,
  and delta size between successive trades

Inputs/Outputs
- Input: A CSV exported by a scraper, e.g., `dom_data_*.csv`
- Output: `dom_ticks_metrics.csv` (consumed by heatmap/3D scripts)

Usage
- Update `input_file` below to point to your latest DOM data CSV
- Run: `python metrics_helper.py`

Dependencies
- pandas
"""

import pandas as pd
import csv
import os

input_file = "dom_data_SPY_20250807_113322.csv"
output_file = "dom_ticks_metrics.csv"

# Check file exists and has at least 1 line of data
if not os.path.exists(input_file):
    raise FileNotFoundError("No dom_ticks.csv found. Ensure your Selenium scraper is running first.")

# Load only the base columns from the streamed file
df = pd.read_csv(input_file, usecols=[
    "timestamp", "symbol", "bid_price", "bid_size",
    "ask_price", "ask_size", "trade_price", "trade_size"
])

# Clean data types
df["bid_price"] = pd.to_numeric(df["bid_price"], errors="coerce")
df["ask_price"] = pd.to_numeric(df["ask_price"], errors="coerce")
df["bid_size"] = pd.to_numeric(df["bid_size"], errors="coerce")
df["ask_size"] = pd.to_numeric(df["ask_size"], errors="coerce")
df["trade_price"] = pd.to_numeric(df["trade_price"], errors="coerce")
df["trade_size"] = pd.to_numeric(df["trade_size"], errors="coerce")

# Drop rows with critical NaNs
df = df.dropna()

# Compute metrics
df["spread"] = df["ask_price"] - df["bid_price"]
df["mid_price"] = (df["ask_price"] + df["bid_price"]) / 2
df["spread_pct"] = df["spread"] / df["mid_price"]
df["imbalance"] = (df["bid_size"] - df["ask_size"]) / (df["bid_size"] + df["ask_size"])
df["trade_direction"] = df["trade_price"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
df["delta_size"] = df["trade_size"].diff().fillna(0)

# Write to new output file
df.to_csv(output_file, index=False)
print(f"✅ Metrics saved to {output_file}")
