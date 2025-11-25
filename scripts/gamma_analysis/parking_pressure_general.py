#!/usr/bin/env python3
"""
parking_pressure_general.py

Generalized version of the gamma-weighted parking pressure model.
Works with ANY ticker's option chain CSV as long as the columns follow:

Strike, Open Interest, Open Interest.1, Gamma, Gamma.1

Usage:
    python parking_pressure_general.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ================================================================
# =============== USER PARAMETERS (EDIT THESE ONLY) ===============
# ================================================================

CSV_PATH = Path(r"F:\inputs\options\05_19_2025\spy_quotedata.csv")

SKIP_ROWS = 3            # First rows to skip before actual chain
PLOT_TITLE = None        # Auto-set from ticker, unless overridden
SAVE_PLOT = True
SAVE_PATH = Path("parking_pressure_curve.png")

LINE_WIDTH = 2.2
GRID_ALPHA = 0.25

# ================================================================
# ======================== LOADING LOGIC ==========================
# ================================================================

def load_chain(csv_path, skip_rows):
    """
    Load option chain CSV, skipping metadata.
    Returns cleaned dataframe with numeric strike, OI, gamma.
    """
    df = pd.read_csv(csv_path, skiprows=skip_rows)

    for col in ["Strike", "Open Interest", "Open Interest.1", "Gamma", "Gamma.1"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Infer ticker from first row of 'Calls' column if possible
    if "Calls" in df.columns:
        try:
            raw = str(df["Calls"].iloc[0])
            ticker = "".join([c for c in raw if c.isalpha()])
        except:
            ticker = "UNKNOWN"
    else:
        ticker = "UNKNOWN"

    return df, ticker


# ================================================================
# ========================= AGGREGATION ===========================
# ================================================================

def aggregate_by_strike(df):
    """
    Aggregate gamma × OI by strike.
    """
    agg = df.groupby("Strike").agg(
        call_gamma_oi=("Gamma", lambda x: np.sum(x * df.loc[x.index, "Open Interest"])),
        put_gamma_oi=("Gamma.1", lambda x: np.sum(x * df.loc[x.index, "Open Interest.1"]))
    ).reset_index()

    return agg.sort_values("Strike").reset_index(drop=True)


# ================================================================
# ===================== PRESSURE CALCULATION ======================
# ================================================================

def compute_parking_pressure(agg):
    """
    Compute downside and upside gamma-weighted OI integrals.
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


# ================================================================
# ============================ PLOT ===============================
# ================================================================

def plot_pressure(df, ticker, title, lw):
    """
    Plot the gamma-weighted pressure curve across *full* strike range.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(df["S_candidate"], df["net_gamma_pressure"], linewidth=lw)

    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.grid(alpha=GRID_ALPHA)

    if title:
        plt.title(title)
    else:
        plt.title(f"{ticker.upper()} Gamma-weighted Parking Pressure")

    plt.xlabel("Candidate Spot Level (Strike)")
    plt.ylabel("Net Gamma-Weighted Put - Call Pressure")

    plt.tight_layout()
    return plt


# ================================================================
# ============================= MAIN ==============================
# ================================================================

def main():
    print(f"\nLoading file: {CSV_PATH}")

    df, inferred_ticker = load_chain(CSV_PATH, SKIP_ROWS)
    ticker = inferred_ticker or "UNKNOWN"
    print(f"Inferred ticker: {ticker}")

    print("Aggregating by strike…")
    agg = aggregate_by_strike(df)

    print("Computing pressure curve…")
    pressure_df = compute_parking_pressure(agg)

    print("Plotting full strike range…")
    plt_obj = plot_pressure(
        pressure_df,
        ticker=ticker,
        title=PLOT_TITLE,
        lw=LINE_WIDTH
    )

    if SAVE_PLOT:
        plt_obj.savefig(SAVE_PATH, dpi=150)
        print(f"Saved gamma-pressure curve to {SAVE_PATH}")

    plt_obj.show()


if __name__ == "__main__":
    main()
