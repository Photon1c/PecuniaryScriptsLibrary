#!/usr/bin/env python3
"""
parking_pressure_curve.py

Standalone script to compute and plot the gamma-weighted
net parking pressure curve for a SPY option chain CSV.

Usage:
    python parking_pressure_curve.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ================================================================
# ===================== USER PARAMETERS ==========================
# ================================================================

# Path to the CSV file
CSV_PATH = Path(r"C:/workingcauldron/inputs/options/log/spy/05_19_2025/spy_quotedata.csv")

# Number of header rows to skip before the option chain begins
SKIP_ROWS = 3

# Range of candidate spot levels to visualize
STRIKE_MIN = 500
STRIKE_MAX = 650

# Plot appearance
PLOT_TITLE = "Gamma-weighted Net Parking Pressure vs Strike"
LINE_WIDTH = 2.0

# Save chart?
SAVE_PLOT = True
SAVE_PATH = Path("parking_pressure_curve.png")

# ================================================================
# ====================== CORE FUNCTIONS ==========================
# ================================================================

def load_chain(csv_path, skip_rows):
    """
    Load SPY option chain CSV, skipping the metadata rows.
    Returns a cleaned DataFrame.
    """
    df = pd.read_csv(csv_path, skiprows=skip_rows)

    # Convert numeric columns safely
    for col in ["Strike", "Open Interest", "Open Interest.1", "Gamma", "Gamma.1"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def aggregate_by_strike(df):
    """
    Group by strike and compute gamma-weighted OI for calls/puts.
    """
    agg = df.groupby("Strike").agg(
        call_gamma_oi=("Gamma", lambda x: np.sum(x * df.loc[x.index, "Open Interest"])),
        put_gamma_oi=("Gamma.1", lambda x: np.sum(x * df.loc[x.index, "Open Interest.1"]))
    ).reset_index()

    agg = agg.sort_values("Strike").reset_index(drop=True)
    return agg


def compute_parking_pressure(agg):
    """
    Compute the gamma-weighted net pressure curve:
    downside = integrated put gamma-OI below S
    upside = integrated call gamma-OI above S
    """
    strikes = agg["Strike"].values
    cg = agg["call_gamma_oi"].values
    pg = agg["put_gamma_oi"].values

    net_pressures = []
    for S in strikes:
        downside = pg[strikes <= S].sum()
        upside = cg[strikes >= S].sum()
        net_pressures.append(downside - upside)

    return pd.DataFrame({
        "S_candidate": strikes,
        "net_gamma_pressure": net_pressures
    })


def plot_pressure(df, smin, smax, title, lw):
    """
    Plot the net parking pressure curve with strike boundaries.
    """
    sub = df[(df["S_candidate"] >= smin) & (df["S_candidate"] <= smax)]

    plt.figure(figsize=(10, 5))
    plt.plot(sub["S_candidate"], sub["net_gamma_pressure"], linewidth=lw)
    plt.axhline(0, linestyle="--", color="black", alpha=0.5)

    plt.title(title)
    plt.xlabel("Candidate Spot Level (S)")
    plt.ylabel("Net Gamma-Weighted Put - Call Pressure")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


# ================================================================
# =========================== MAIN ===============================
# ================================================================

def main():
    print(f"\nLoading CSV: {CSV_PATH}")
    df = load_chain(CSV_PATH, SKIP_ROWS)

    print("Aggregating by strike...")
    agg = aggregate_by_strike(df)

    print("Computing parking pressure curve...")
    pressure_df = compute_parking_pressure(agg)

    print("Plotting...")
    plt_obj = plot_pressure(
        pressure_df, STRIKE_MIN, STRIKE_MAX,
        title=PLOT_TITLE,
        lw=LINE_WIDTH
    )

    if SAVE_PLOT:
        plt_obj.savefig(SAVE_PATH, dpi=150)
        print(f"Saved plot to {SAVE_PATH}")

    plt_obj.show()


if __name__ == "__main__":
    main()
