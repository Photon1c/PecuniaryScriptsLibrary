"""
Advanced Metrics Helper for DOM Ticks
=====================================

Purpose
- Derive higher-level indicators from streamed DOM ticks with metrics.

What it does
- Aggregates trade size per second and counts price diversity
- Computes a simple iceberg likelihood score
- Inspects pinned price conditions across bid/ask/mid sequences

Inputs/Outputs
- Input: `dom_ticks_metrics.csv`
- Output: `analyze_dom_ticks.csv` plus console summaries

Usage
- `python adv_metrics_helper.py`

Dependencies
- pandas, numpy
"""
import pandas as pd
import numpy as np

# Load the streamed tick data with computed metrics
df = pd.read_csv("dom_ticks_metrics.csv")

# Ensure timestamp is datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Round timestamp to the second
df['second'] = df['timestamp'].dt.floor('S')

# --- 1. Cumulative trade size per second ---
cumulative_df = df.groupby('second').agg({
    'trade_size': 'sum',
    'trade_price': ['nunique', 'count']
})
cumulative_df.columns = ['cum_trade_size', 'unique_trade_prices', 'num_trades']
cumulative_df.reset_index(inplace=True)

# --- 2. Iceberg likelihood score ---
# If multiple trades at same price in a second with small sizes, increase iceberg score
# Score: +1 for repeated small trades at same price (<100 shares per trade)
def compute_iceberg_score(group):
    sizes = group['trade_size']
    prices = group['trade_price']
    same_price = prices.nunique() == 1
    small_trades = (sizes < 100).sum()
    return int(same_price and small_trades >= 3)

iceberg_scores = df.groupby('second').apply(compute_iceberg_score).reset_index(name='iceberg_score')

# --- 3. Price pinning indicator ---
# If bid, ask, and mid_price stay unchanged over multiple ticks
df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2
df['pin_check'] = df[['bid_price', 'ask_price', 'mid_price']].astype(str).agg('-'.join, axis=1)
pin_counts = df.groupby('pin_check').size().reset_index(name='pin_count')
most_pinned = pin_counts.sort_values(by='pin_count', ascending=False).head(1)

# Merge results
final_df = cumulative_df.merge(iceberg_scores, on='second')
final_df.to_csv("analyze_dom_ticks.csv")

print(final_df)
print("Most pinned: ")
print(most_pinned)