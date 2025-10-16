# Quantum Tunneling Zones Visualization

## Overview

The Quantum Tunneling model now includes an interactive HTML dashboard that visualizes breakout probabilities across option strikes.

## Features

### 📊 Two Interactive Charts

1. **Stacked Bar Chart**: Shows tunneling probability by strike, grouped by calls/puts
2. **Scatter/Strip Plot**: Distribution view with volume-weighted sizing

### 🎨 Color-Coded Zones

- **🔵 LOW** (#4C78A8): Probability < 0.25
- **🟠 MEDIUM** (#F58518): Probability 0.25 - 0.50
- **🔴 HIGH** (#E45756): Probability ≥ 0.50

### 🖱️ Interactive Features

- **Hover tooltips** show:
  - Strike, Probability, Zone
  - Mid, Bid, Ask prices
  - Spread, IV, OI, Volume
- **Zoom/Pan** enabled via Plotly controls
- **Threshold lines** at 0.25 and 0.50

## Usage

### Basic Run

```bash
cd crimson_switchblade/quantum_tunneling
python run_analysis.py
```

This generates:
- `tunneling_report.csv` - Tabular data
- `tunneling_report.html` - Interactive visualization (opens offline in any browser)

### Customization

#### Adjust Thresholds via Environment Variables

```bash
# Set custom thresholds
export QT_LOW=0.20
export QT_HIGH=0.50
python run_analysis.py
```

#### Modify Center Strike

Edit `run_analysis.py`:
```python
CENTER_STRIKE = 665.0  # Your preferred strike
WINDOW = 10.0          # +/- range
```

## Output Files

### tunneling_report.csv
Tabular data with columns:
- `strike_k`, `opt_side`, `mid`, `bid`, `ask`
- `spread`, `volume_x`, `oi_x`, `iv_x`, `gamma_x`
- `imbalance`, `barrier_strength`
- `tunnel_prob`, `tunnel_flag`

### tunneling_report.html
Standalone HTML dashboard with:
- No server required
- Works offline
- Self-contained (CDN for Plotly.js only)
- Modern responsive design

## Understanding the Model

### Tunneling Probability Formula

```
P(tunnel) = exp(-κ × barrier_strength / (|imbalance| + 1))
```

Where:
- **κ (kappa)**: Scaling factor (default 500.0)
- **barrier_strength**: Function of spread, liquidity, IV
- **imbalance**: Call volume - Put volume at strike

### Barrier Strength Factors

Higher probability (easier tunneling) when:
- ✅ Wide bid-ask spreads
- ✅ Low volume and open interest
- ✅ Strong directional imbalance
- ✅ IV compression

Lower probability (harder tunneling) when:
- ❌ Tight spreads
- ❌ High liquidity (volume + OI)
- ❌ Balanced call/put pressure
- ❌ Elevated IV

## File Structure

```
crimson_switchblade/quantum_tunneling/
├── quantum_tunneling_model.py  # Core calculation engine
├── plot_utils.py                # NEW: Visualization module
├── run_analysis.py              # UPDATED: Main runner
├── tunneling_report.csv         # Generated: Data
└── tunneling_report.html        # Generated: Dashboard
```

## Dependencies

```bash
pip install pandas plotly
```

No other dependencies required!

## Advanced Usage

### Programmatic Access

```python
from pathlib import Path
from quantum_tunneling_model import compute_metrics, focus_window
from plot_utils import render_tunneling_html
import pandas as pd

# Load and process
df = pd.read_csv("your_data.csv", skiprows=3)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
result = compute_metrics(df)
focused = focus_window(result, center_strike=665, width=10)

# Generate visualization
render_tunneling_html(
    focused,
    Path("my_report.html"),
    title="Custom Analysis",
    kappa=500.0
)
```

### Missing Column Handling

The visualization gracefully handles missing columns:
- If `opt_side` missing → Single bar series
- If `volume_x` missing → Strip plot instead of sized scatter
- If Greeks missing → Omitted from hover tooltips

## Troubleshooting

### HTML Not Generated

Check:
1. Plotly installed: `pip install plotly`
2. Write permissions in output directory
3. Check console for `[WARN]` messages

### Incorrect Colors

Verify thresholds:
```python
import os
print(f"LOW: {os.getenv('QT_LOW', '0.25')}")
print(f"HIGH: {os.getenv('QT_HIGH', '0.50')}")
```

### Empty Charts

Ensure data has required columns:
```python
required = ['strike_k', 'tunnel_prob', 'tunnel_flag']
missing = [c for c in required if c not in df.columns]
if missing:
    print(f"Missing columns: {missing}")
```

## Examples

### Example 1: SPY Weekly Options
```bash
# In run_analysis.py
CENTER_STRIKE = 575.0
WINDOW = 5.0
PREFERRED_EXPIRY = "2025-10-18"

python run_analysis.py
# → Open tunneling_report.html
```

### Example 2: High Kappa (More Selective)
```python
# In plot_utils.py or pass as parameter
render_tunneling_html(df, path, kappa=1000.0)
# → Fewer HIGH zones, more conservative
```

### Example 3: Custom Thresholds
```bash
export QT_LOW=0.15
export QT_HIGH=0.60
python run_analysis.py
# → More MEDIUM zone spread
```

## Best Practices

1. **Update Data Regularly**: Options data gets stale quickly
2. **Check Multiple Expirations**: Different DTE = different dynamics
3. **Cross-Reference Volume**: High tunnel prob + high volume = stronger signal
4. **Watch Imbalance**: Large imbalances indicate directional conviction
5. **Monitor IV Changes**: Compression/expansion affects barrier strength

## Future Enhancements

Potential additions (not yet implemented):
- [ ] Time-series animation of tunneling evolution
- [ ] Multi-expiry overlay comparison
- [ ] Correlation with actual price moves
- [ ] Real-time updating dashboard
- [ ] ML-based threshold optimization

---

**Built for options traders who think in probabilities, not certainties.** 🌊

*Note: This is a research tool. Past patterns don't guarantee future outcomes.*

