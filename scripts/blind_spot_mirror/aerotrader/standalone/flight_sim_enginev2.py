# flight_sim_engine.py
# Generates report, but uses mock data
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

# --- Compute Dummy Intraday Metrics (TODO: Replace with real logic) ---
# For now, we'll sample 5 timestamps and simulate gain %, fuel %, IV spike, stall
timestamps = ["09:30", "10:30", "12:00", "13:30", "16:00"]
price_changes = [0.0, 1.2, 3.9, 5.1, 4.7]  # Dummy gain %
fuel_levels = [100, 75, 50, 35, 0]         # Dummy fuel depletion
iv_spikes = [False, False, True, False, False]
ema_drag = [False, False, True, False, False]

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
