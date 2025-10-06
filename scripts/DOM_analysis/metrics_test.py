"""
Metrics Test Generator
======================

Purpose
- Produce a small `dom_ticks.csv` with both base fields and computed metrics
  to validate downstream processing and charting.

What it does
- Appends a few synthetic rows, computing spread, mid price, imbalance,
  trade direction, and delta size.

Usage
- `python metrics_test.py`

Outputs
- Appends to or creates `dom_ticks.csv`; prints the last few rows for review.
"""
import os
import csv
from datetime import datetime
import pandas as pd

# Setup CSV file path
output_file = "dom_ticks.csv"

# Define full header including both base and computed fields
headers = [
    "timestamp", "symbol", "bid_price", "bid_size",
    "ask_price", "ask_size", "trade_price", "trade_size",
    "spread", "mid_price", "spread_pct", "imbalance",
    "trade_direction", "delta_size"
]

# Initialize file with header if not present
if not os.path.exists(output_file):
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

# Sample data (your real scraper will fill first 8 fields)
sample_ticks = [
    ['2025-08-04 11:21:36', 'SPY', '630.05', '300', '630.10', '1450', '630.09', '80'],
    ['2025-08-04 11:21:38', 'SPY', '630.05', '400', '630.10', '1570', '630.08', '1'],
    ['2025-08-04 11:21:40', 'SPY', '630.05', '400', '630.10', '1570', '630.08', '1'],
    ['2025-08-04 11:21:42', 'SPY', '630.05', '250', '630.10', '1550', '630.06', '86'],
]

prev_mid_price = None
prev_trade_size = None

# Append with computed metrics
with open(output_file, mode='a', newline='') as f:
    writer = csv.writer(f)
    for tick in sample_ticks:
        timestamp, symbol, bid_price, bid_size, ask_price, ask_size, trade_price, trade_size = tick
        bid_price = float(bid_price)
        ask_price = float(ask_price)
        bid_size = int(bid_size.replace(',', ''))
        ask_size = int(ask_size.replace(',', ''))
        trade_price = float(trade_price)
        trade_size = int(trade_size.replace(',', ''))

        spread = ask_price - bid_price
        mid_price = (ask_price + bid_price) / 2
        spread_pct = spread / mid_price if mid_price else 0
        imbalance = (bid_size - ask_size) / (bid_size + ask_size) if (bid_size + ask_size) else 0
        trade_direction = (
            1 if prev_mid_price is not None and trade_price > prev_mid_price else
            -1 if prev_mid_price is not None and trade_price < prev_mid_price else 0
        )
        delta_size = trade_size - prev_trade_size if prev_trade_size is not None else 0

        writer.writerow([
            timestamp, symbol, bid_price, bid_size, ask_price, ask_size,
            trade_price, trade_size, spread, mid_price, spread_pct,
            imbalance, trade_direction, delta_size
        ])

        prev_mid_price = mid_price
        prev_trade_size = trade_size

# ✅ Read and print for validation
df = pd.read_csv(output_file)
print(df.tail(10))
