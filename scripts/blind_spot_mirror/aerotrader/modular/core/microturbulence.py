# microturbulence.py
"""
Estimates a mock intraday IV curve using recent IV delta.
For now, returns 5 random values based on last known IV.
"""

import random

def estimate_intraday_iv(option_df) -> list:
    latest_iv = option_df["IV"].dropna().astype(float).mean()
    return [round(latest_iv + random.uniform(-0.1, 0.1), 2) for _ in range(5)]
