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
MARKDOWN_OUTPUT_PATH = f"logs/flight_log_{TICKER.lower()}.md"

# --- Load Data ---
stock_df = load_stock_data(TICKER)
option_df = load_option_data(TICKER)

# --- Process Stock Data ---
stock_df['Date'] = pd.to_datetime(stock_df['Date'])
stock_df = stock_df.sort_values('Date')
stock_df['Close/Last'] = pd.to_numeric(stock_df['Close/Last'], errors='coerce')

# --- Generate synthetic intraday times for the most recent date ---
latest_date = stock_df['Date'].max().date()
intraday_times = ["09:30", "10:30", "12:00", "13:30", "16:00"]
timestamps = [f"{latest_date} {t}" for t in intraday_times]

# For now, we'll sample 5 evenly spaced rows from the most recent date's data (or use mock data)
sampled = stock_df[stock_df['Date'].dt.date == latest_date].copy()
if sampled.empty or len(sampled) < len(intraday_times):
    # Use mock data if not enough real data for the latest date
    price_changes = [0.0, 1.2, 3.9, 5.1, 4.7]  # Dummy gain %
    fuel_levels = [100, 75, 50, 35, 0]         # Dummy fuel depletion
    iv_spikes = [False, False, True, False, False]
    ema_drag = [False, False, True, False, False]
else:
    # Use real data for the latest date, resampled to match intraday_times
    sampled = sampled.iloc[::max(1, len(sampled)//len(intraday_times))].copy()
    sampled = sampled.head(len(intraday_times))
    open_price = sampled['Close/Last'].iloc[0]
    price_changes = ((sampled['Close/Last'] - open_price) / open_price * 100).round(2).tolist()
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
    ema_drag = (abs(sampled['Close/Last'] - sampled['EMA20']) < 0.5).tolist()

# --- Helpers ---
def detect_turbulence(iv_spike): return "Heavy" if iv_spike else "Light"
def detect_stall(drag): return "Yes" if drag else "No"
def get_status(i, total):
    if i == 0: return "Takeoff"
    if i == total - 1: return "Landed"
    if price_changes[i] - price_changes[i-1] < -1: return "Emergency descent"
    return "Cruising altitude"

# --- Markdown Generator ---
def generate_flight_log_md():
    with open(MARKDOWN_OUTPUT_PATH, "w", encoding="utf-8") as f:

        f.write(f"# ✈️ Flight Summary - {TICKER.upper()} - {datetime.today().strftime('%m/%d/%Y')}\n")
        f.write(f"- Net Gain: {price_changes[-1]:+.2f}%\n")
        f.write(f"- Max Altitude: {max(price_changes):+.2f}%\n")
        f.write(f"- Fuel Remaining: {fuel_levels[-1]}%\n")
        f.write(f"- Stall Events: {sum(ema_drag)}\n")
        f.write(f"- Emergency Landings: 0\n\n")

        f.write("| Time              | Altitude (%) | Fuel (%) | Stall | Turbulence | Status           |\n")
        f.write("|-------------------|--------------|----------|-------|------------|------------------|\n")
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
