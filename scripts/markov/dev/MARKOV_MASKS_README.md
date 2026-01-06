# Markov Masks: Contract-Level Agency Detection

## Overview

The Markov Masks module extends the Markov Blanket analysis to the contract level, encoding structural position, volatility geometry, and sensitivity profiles to detect local agency transfer and expressivity.

## New Files

### `markov_mask.py`
New module containing the `MarkovMaskComputer` class with methods:
- `compute_structural_position()`: dS, abs_dS, log_moneyness, dFlip, wall_distance, moneyness_bucket
- `compute_volatility_geometry()`: IV level, IV slope, IV curvature, term slope, local skew
- `compute_sensitivity_profile()`: delta, gamma, vega, theta, rho (from existing columns)
- `compute_local_skew()`: put IV - call IV at matching strikes
- `compute_expressivity_score()`: Weighted score [0, 1] combining proximity, gamma, vega, curvature, instability, execution quality
- `assign_mask_state()`: Categorizes contracts as DORMANT, SENSITIVE, EXPRESSIVE, RUPTURE_CANDIDATE, or TOXIC
- `compute_masks()`: Main entry point returning DataFrame and summary

## Integration into testB.py

### Changes Made
1. **Import**: Added `from markov_mask import MarkovMaskComputer`
2. **New Section**: Added ">>> Markov Masks (Contract-Level Agency)" after Kelly Gate
3. **DataFrame Preparation**: Extracts calls and puts from option_df (handles missing columns gracefully)
4. **Output Files**:
   - `markov_masks.csv`: One row per contract with all mask features
   - `markov_masks.json`: Metadata, global summary, top lists, mask state counts
5. **Terminal Output**: 
   - Table of top 10 expressive contracts
   - Mask state distribution table
   - Teixiptla interpretation text

## Output Files

### `markov_masks.csv`
Columns:
- `contract_type`: CALL (or PUT if extended)
- `strike`, `expiration`, `premium_mid`
- `iv`, `delta`, `gamma`, `vega`, `theta`
- `dS`, `abs_dS`, `log_moneyness`, `abs_dFlip`, `wall_distance`
- `moneyness_bucket`: ITM/ATM/OTM
- `iv_slope_local`, `iv_curvature_local`, `term_slope`, `skew_local`
- `expressivity_score`: [0, 1]
- `mask_state`: DORMANT/SENSITIVE/EXPRESSIVE/RUPTURE_CANDIDATE/TOXIC

### `markov_masks.json`
Structure:
```json
{
  "run_metadata": {
    "ticker": "SPY",
    "timestamp": "2025-12-27T23:56:00",
    "spot_price": 690.31,
    "output_folder": "../output/analysis_..."
  },
  "global_blanket_summary": {
    "residual_r2": 0.148,
    "skew_r2": 0.334,
    "extra_pct": 1.4,
    "regime": "PIN",
    "structure_family": "PROBE_ONLY",
    "gate_state": "BLOCK"
  },
  "top_masks": {
    "top_by_expressivity": [...],
    "top_by_gamma": [...],
    "top_by_vega": [...],
    "top_by_curvature": [...],
    "top_by_execution_quality": [...],
    "recommended_watchlist": [...]
  },
  "mask_state_counts": {
    "DORMANT": 850,
    "SENSITIVE": 200,
    "EXPRESSIVE": 50,
    "RUPTURE_CANDIDATE": 10,
    "TOXIC": 28
  }
}
```

## Expressivity Score Components

Weighted combination (tunable via `EXPRESSIVITY_WEIGHTS`):
- **Proximity to spot** (20%): Higher when close to ATM
- **Gamma strength** (25%): Normalized |gamma|
- **Vega strength** (20%): Normalized vega
- **Curvature signal** (15%): Normalized |IV curvature|
- **Instability gate** (10%): Higher in TRANSFER regime, boosted for CONVEXITY
- **Execution quality** (10%): Reuses execution quality score

## Mask States

- **DORMANT**: Low gamma/vega, far from spot
- **SENSITIVE**: Near spot, moderate gamma/vega, low curvature
- **EXPRESSIVE**: High gamma/vega, strong curvature
- **RUPTURE_CANDIDATE**: EXPRESSIVE + expressivity > 0.7
- **TOXIC**: Poor execution quality (spread > 15% or quality < 0.3)

## Sample Terminal Output

```
>>> Markov Masks (Contract-Level Agency)
Markov Masks CSV saved: ../output/analysis_.../markov_masks.csv
Markov Masks JSON saved: ../output/analysis_.../markov_masks.json

        Top 10 Expressive Contracts (Markov Masks)
┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Strike ┃ Expiry     ┃ Type ┃ Premium ┃ IV    ┃ Delta ┃ Gamma  ┃ Vega  ┃ Theta  ┃ Mask State ┃ Expressivity┃
┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ $690.00│ 2025-01-17 │ CALL │ $2.45   │ 18.5% │ 0.523 │ 0.0123 │ 0.234 │ -0.045 │ EXPRESSIVE │ 0.782       │
│ ...    │ ...        │ ...  │ ...     │ ...   │ ...   │ ...    │ ...   │ ...    │ ...        │ ...         │
└────────┴────────────┴──────┴─────────┴───────┴───────┴────────┴───────┴────────┴────────────┴─────────────┘

        Mask State Distribution
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━┓
┃ Mask State           ┃ Count ┃ Percentage┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━┩
│ DORMANT              │ 850   │ 74.7%    │
│ SENSITIVE            │ 200   │ 17.6%    │
│ EXPRESSIVE           │ 50    │ 4.4%     │
│ TOXIC                │ 28    │ 2.5%     │
│ RUPTURE_CANDIDATE    │ 10    │ 0.9%     │
└──────────────────────┴───────┴──────────┘

Teixiptla Interpretation:
Markov masks encode contract-level agency transfer. 1138 contracts analyzed.
Recommended watchlist: 8 contracts (high expressivity, non-toxic execution).
Mask states indicate local expressivity: EXPRESSIVE/RUPTURE_CANDIDATE contracts become expressive first when convexity is permitted.
```

## Configuration

### Expressivity Weights (in `markov_mask.py`)
```python
EXPRESSIVITY_WEIGHTS = {
    'proximity_to_spot': 0.20,
    'gamma_strength': 0.25,
    'vega_strength': 0.20,
    'curvature_signal': 0.15,
    'instability_gate': 0.10,
    'execution_quality': 0.10
}
```

### Mask State Thresholds (in `markov_mask.py`)
```python
THRESHOLDS = {
    'dormant_gamma': 0.01,
    'dormant_vega': 0.05,
    'dormant_distance': 0.10,
    'sensitive_distance': 0.05,
    'sensitive_gamma': 0.05,
    'expressive_gamma': 0.10,
    'expressive_vega': 0.20,
    'expressive_curvature': 0.01,
    'toxic_spread_pct': 15.0,
    'toxic_quality': 0.3
}
```

### IV Regression Window
- Default: 5 strikes (configurable via `IV_REGRESSION_WINDOW`)

## Features

### Structural Position
- **dS**: Distance to spot (spot - strike)
- **abs_dS**: Absolute distance
- **log_moneyness**: log(strike/spot)
- **dFlip**: Distance to gamma flip level (if provided)
- **wall_distance**: Distance to nearest dealer wall (if provided)
- **moneyness_bucket**: ITM/ATM/OTM classification

### Volatility Geometry
- **iv_local**: Contract IV
- **iv_slope_local**: Linear fit of IV vs strike (windowed)
- **iv_curvature_local**: Quadratic coefficient (windowed)
- **term_slope**: IV vs days-to-expiry slope
- **skew_local**: Put IV - Call IV at matching strikes

### Sensitivity Profile
- Extracted from existing columns: Delta, Gamma, Vega, Theta, Rho
- Falls back to NaN if columns missing (doesn't break)

## Error Handling

- Graceful degradation: Missing columns → NaN
- Empty arrays handled safely
- Try-except blocks around all computations
- Safe defaults returned on errors
- Logs warnings but doesn't crash

## Performance

- Vectorized operations where possible
- Windowed regression precomputed per expiry
- Efficient strike matching for local skew
- Reasonable runtime (adds ~1-2 seconds typically)

## Future Enhancements

- Add flip_level detection (gamma flip calculation)
- Add wall detection (call/put concentration analysis)
- Extend to puts (currently calls only)
- Add per-contract execution quality (currently aggregated)
- Add visualization plots for mask states

## Usage

The module is automatically called during testB.py execution. No additional flags needed.

```powershell
python testB.py --ticker SPY
```

Outputs will be in the same dated folder as other results:
- `markov_masks.csv`
- `markov_masks.json`
