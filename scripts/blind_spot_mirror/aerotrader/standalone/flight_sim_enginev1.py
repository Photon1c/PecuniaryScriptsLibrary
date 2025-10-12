# flight_sim_engine.py
# Simple first version that uses mock data
"""
✈️ Flight Simulation Engine for Daily Trading Sessions
-----------------------------------------------------
Simulates a flight based on intraday SPY price action.
Outputs a GitHub-compatible Markdown dashboard with:

- Altitude (profit %)
- Fuel (liquidity %)
- Stall and turbulence warnings
- Flight status (takeoff, cruising, landing)

TODO: Replace dummy data sections with real CSV or API inputs
"""

import pandas as pd
from datetime import datetime

# --- Dummy Inputs (Replace with actual loaders) ---
# Simulated 5 timepoints during a mock trading day
timestamps = ["09:30", "10:30", "12:00", "13:30", "16:00"]
price_changes = [0.0, 1.2, 3.9, 5.1, 4.7]  # % gain from open
fuel_levels = [100, 75, 50, 35, 0]         # % liquidity remaining
iv_spikes = [False, False, True, False, False]  # turbulence
ema_drag = [False, False, True, False, False]   # stalling

# --- Helper Functions ---

def detect_turbulence(iv_spike):
    return "Heavy" if iv_spike else "Light"

def detect_stall(drag):
    return "Yes" if drag else "No"

def get_status(i, total):
    if i == 0:
        return "Takeoff"
    elif i == total - 1:
        return "Landed"
    elif price_changes[i] - price_changes[i-1] < -1:
        return "Emergency descent"
    else:
        return "Cruising altitude"

# --- Markdown Table Generator ---

def generate_flight_log():
    print("# ✈️ Flight Summary - SPY - 06/26/2025")
    print(f"- Net Gain: {price_changes[-1]:+.2f}%")
    print(f"- Max Altitude: {max(price_changes):+.2f}%")
    print(f"- Fuel Remaining: {fuel_levels[-1]}%")
    print(f"- Stall Events: {sum(ema_drag)}")
    print(f"- Emergency Landings: 0\n")

    print("| Time  | Altitude (%) | Fuel (%) | Stall | Turbulence | Status           |")
    print("|-------|---------------|----------|--------|------------|------------------|")
    for i in range(len(timestamps)):
        row = [
            timestamps[i],
            f"{price_changes[i]:+.2f}%",
            f"{fuel_levels[i]}%",
            detect_stall(ema_drag[i]),
            detect_turbulence(iv_spikes[i]),
            get_status(i, len(timestamps))
        ]
        print("| " + " | ".join(row) + " |")

# --- Main Entry ---

if __name__ == "__main__":
    generate_flight_log()
