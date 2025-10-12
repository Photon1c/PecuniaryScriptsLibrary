# intraday_emulator.py
"""
Simulates a 5-step intraday flight path from a single OHLC daily candle:
- Open
- Mid-AM
- Midday
- Mid-PM
- Close

Uses candle shape to interpolate plausible price progression.
"""

import pandas as pd

def simulate_intraday_path(row: pd.Series) -> dict:
    open_ = row["Open"]
    high = row["High"]
    low = row["Low"]
    close = row["Close/Last"]

    # Midpoints for staged path
    # Basic assumption: prices rise toward high and fall toward low at some point
    path = {
        "09:30": open_,
        "10:30": (open_ + high) / 2,
        "12:00": (high + low) / 2,
        "13:30": (low + close) / 2,
        "16:00": close,
    }

    # Convert to % gain from open
    open_price = open_
    gain_series = {t: round((p - open_price) / open_price * 100, 2) for t, p in path.items()}
    return gain_series
