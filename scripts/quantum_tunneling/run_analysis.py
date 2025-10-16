
"""
run_analysis.py
---------------
Example runner that loads a CSV, computes tunneling metrics, and writes a small report.
"""
import pandas as pd
from pathlib import Path
from quantum_tunneling_model import compute_metrics, focus_window
from plot_utils import render_tunneling_html

IN_PATH = Path("F:/inputs/options/log/spy/10_13_2025/spy_quotedata.csv")
OUT_PATH = Path("tunneling_report.csv")

CENTER_STRIKE = 665.0    # focus near your trade
WINDOW = 10.0            # +/- strikes
PREFERRED_EXPIRY = None  # e.g., "2025-10-18" if you want a specific expiry

df = pd.read_csv(IN_PATH, skiprows=3)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
res = compute_metrics(df)

focus = focus_window(res, CENTER_STRIKE, WINDOW, PREFERRED_EXPIRY)

# Select compact columns for the report
keep = ["strike_k","opt_side","mid","bid","ask","spread","volume_x","oi_x","iv_x","gamma_x","imbalance","barrier_strength","tunnel_prob","tunnel_flag"]
present = [c for c in keep if c in focus.columns]
report = focus[present].sort_values(["tunnel_flag","tunnel_prob"], ascending=[True, False])

report.to_csv(OUT_PATH, index=False)
print(f"Wrote: {OUT_PATH}")

# Generate HTML visualization
HTML_PATH = OUT_PATH.with_suffix(".html")  # Same folder as CSV, just .html extension
try:
    render_tunneling_html(
        focus, 
        HTML_PATH, 
        title=f"SPY Quantum Tunneling Zones (center {CENTER_STRIKE:.0f})",
        kappa=500.0  # Match the kappa value from quantum_tunneling_model.py
    )
    print(f"Wrote: {HTML_PATH}")
except Exception as e:
    print(f"[WARN] Could not render HTML: {e}")
