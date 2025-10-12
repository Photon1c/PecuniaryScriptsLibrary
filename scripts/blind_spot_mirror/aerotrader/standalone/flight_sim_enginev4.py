# flight_sim_engine.py
# Adds report, but still uses mock data
"""
✈️ Flight Simulation Engine for Daily Trading Sessions

This script simulates a flight for a given ticker (default: SPY) using real data
loaded via your `data_loader` module. Outputs a Markdown flight log.

Run it as a CLI tool or inside a notebook.
"""

import pandas as pd
from data_loader import load_stock_data, load_option_data
from datetime import datetime

# --- CONFIG ---
TICKER = "SPY"
MARKDOWN_OUTPUT_PATH = f"flight_log_{TICKER.lower()}.md"

# --- Load Data ---
stock_df = load_stock_data(TICKER)
option_df = load_option_data(TICKER)

# --- Process Stock Data ---
stock_df['Date'] = pd.to_datetime(stock_df['Date'])
stock_df = stock_df.sort_values('Date')
stock_df['Close/Last'] = pd.to_numeric(stock_df['Close/Last'], errors='coerce')

# Resample or sample at 5 evenly spaced rows
#sampled = stock_df.iloc[::max(1, len(stock_df)//5)].copy()
#sampled = stock_df.iloc[::max(1, len(stock_df)//6)].copy()
sampled = stock_df.tail(6).iloc[:-1].copy()

sampled.reset_index(drop=True, inplace=True)

open_price = sampled['Close/Last'].iloc[0]
price_changes = ((sampled['Close/Last'] - open_price) / open_price * 100).round(2).tolist()


if sampled['Date'].dt.hour.nunique() > 1:
    timestamps = sampled['Date'].dt.strftime("%H:%M").tolist()
else:
    timestamps = sampled['Date'].dt.strftime("%m/%d").tolist()


volumes = sampled['Volume'].fillna(0)
total_volume = volumes.sum()
fuel_levels = ((total_volume - volumes.cumsum()) / total_volume * 100).round(0).tolist()


option_df = option_df.dropna(subset=['IV'])
iv_series = pd.to_numeric(option_df['IV'], errors='coerce').dropna()

iv_threshold = iv_series.mean() + iv_series.std()
iv_spikes = [(iv_series.sample(1).iloc[0] > iv_threshold) for _ in range(len(sampled))]

# Simple EMA divergence detection
stock_df['EMA20'] = stock_df['Close/Last'].ewm(span=20).mean()
sampled['EMA20'] = stock_df['EMA20'].reindex(sampled.index).values
is_intraday = sampled['Date'].dt.hour.nunique() > 1

ema_drag = (abs(sampled['Close/Last'] - sampled['EMA20']) < 0.5).tolist()


# --- Helpers ---
def detect_turbulence(iv_spike): return "Heavy" if iv_spike else "Light"
def detect_stall(drag): return "Yes" if drag else "No"
def get_status(i, total):
    if i == 0:
        return "Takeoff"
    if i == total - 1:
        return "Landed"

    drop = price_changes[i] - price_changes[i-1]
    fuel = fuel_levels[i]

    if fuel < 10 and price_changes[i] < -5:
        return "Go-around pattern"
    if drop < -1:
        return "Emergency descent"

    return "Cruising altitude"


# --- Markdown Generator ---
def generate_flight_log_md():

    with open(MARKDOWN_OUTPUT_PATH, "w", encoding="utf-8") as f:

        f.write(f"# ✈️ Flight Summary - {TICKER.upper()} - {datetime.today().strftime('%m/%d/%Y')}\n")
        f.write(f"- Mode: {'Jet Flight (Intraday)' if is_intraday else 'Macro Cruise (Daily)'}\n")
        f.write(f"- Net Gain: {price_changes[-1]:+.2f}%\n")
        f.write(f"- Max Altitude: {max(price_changes):+.2f}%\n")
        f.write(f"- Fuel Remaining: {fuel_levels[-1]}%\n")
        f.write(f"- Stall Events: {sum(ema_drag)}\n")
        f.write(f"- Emergency Landings: 0\n\n")

        f.write("| Time  | Altitude (%) | Fuel (%) | Stall | Turbulence | Status           |\n")
        f.write("|-------|---------------|----------|--------|------------|------------------|\n")
        for i in range(len(timestamps)):
            row = [
                timestamps[i],
                f"{price_changes[i]:+.2f}%",
                f"{fuel_levels[i]}%",
                detect_stall(ema_drag[i]),
                detect_turbulence(iv_spikes[i]),
                get_status(i, len(timestamps))
            ]
            f.write("| " + " | ".join(row) + " |\n")

    print(f"[✓] Flight log saved to: {MARKDOWN_OUTPUT_PATH}")

# --- Entry Point ---
if __name__ == "__main__":
    generate_flight_log_md()
