import sys
import os

DATA_ROOT = r"F:\inputs"            # your convention
# Path to aerotrader entry point
AEROTRADER_DIR = os.path.join(os.path.dirname(__file__), "aerotrader", "modular")
AEROTRADER_CMD = [sys.executable, "entry.py"]  # Run via Python
POLL_SECS = 15
MODEL = "gpt-4o-mini"               # swap as you wish

THRESHOLDS = {
    "max_spread_abs": 0.10,
    "min_oi": 300,
    "ivjump_pp_5m": 8.0,
    "rr_min": 1.8
}
