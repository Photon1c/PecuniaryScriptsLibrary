# Teixiptla-Garage-Markov Kelly Gate - Implementation Summary

## Overview

The Kelly Gate module has been integrated into `testB.py` as a permission layer for option trading. It provides:
- **Regime inference**: PIN / PRE_TRANSFER / TRANSFER
- **Structure family recommendation**: MEAN_REVERSION_PREMIUM / CONVEXITY / PROBE_ONLY
- **Kelly sizing**: Fractional Kelly with multipliers based on market structure
- **Gate state**: BLOCK / PROBE / DEPLOY

## New Features

### 1. Skew Features
- Put-call IV difference at ATM
- Skew slope (linear fit of IV vs moneyness for puts/calls)
- Smile curvature (2nd-order polynomial coefficient)
- ATM and 25Δ IV proxies

### 2. Term Structure Features
- Front IV (nearest expiry, DTE < 30)
- Back IV (far expiry, DTE > 60)
- Term slope (IV vs DTE linear fit)
- Inversion flag (front_iv > back_iv by threshold)

### 3. Execution Quality
- Bid-ask spread percentage
- Entry/exit slippage estimation
- Quality score (0-1) based on spread, volume, and open interest

### 4. Kelly Gate Logic
- **Regime inference**: Based on residual R², skew R², and model confidence
- **p, b estimation**: Conservative heuristics using mispricing signals
- **Multipliers**: Applied based on term inversion, skew steepness, execution quality, and regime
- **Gate state**: Determined by adjusted Kelly size and execution quality

## How to Run

### Basic Usage
```powershell
python testB.py --ticker SPY
```

### With Debug Output
```powershell
python testB.py --ticker SPY --debug
```

### Use Raw Premium (disable log transformation)
```powershell
python testB.py --ticker SPY --raw-premium
```

### Custom Output Directory
```powershell
python testB.py --ticker SPY --output ./my_results
```

### Skip SHAP Plots (faster)
```powershell
python testB.py --ticker SPY --skip-shap
```

## Output Files

### 1. `gate.json`
Contains all Kelly Gate results:
```json
{
  "regime": "TRANSFER",
  "structure_family": "CONVEXITY",
  "kelly_raw": 0.0234,
  "kelly_fractional": 0.0059,
  "kelly_adjusted": 0.0042,
  "gate_state": "DEPLOY",
  "p": 0.587,
  "b": 2.0,
  "multiplier": 0.712,
  "skew_features": {
    "put_call_iv_diff": 0.0234,
    "skew_slope_puts": 0.145,
    "skew_slope_calls": -0.089,
    "smile_curvature": 0.012,
    ...
  },
  "term_features": {
    "front_iv": 0.18,
    "back_iv": 0.16,
    "term_slope": -0.0002,
    "is_inverted": true,
    ...
  },
  "execution_quality": {
    "bid_ask_spread_pct": 8.5,
    "quality_score": 0.72,
    ...
  }
}
```

### 2. `contract_scores.csv`
Per-contract execution quality and mispricing metrics:
- `contract_idx`: Contract index
- `strike`: Strike price
- `moneyness`: K/S ratio
- `moneyness_bucket`: ITM/ATM/OTM
- `classical_pred`: Classical model prediction
- `combined_pred`: Combined model prediction
- `mispricing_magnitude`: |combined - classical|
- `execution_quality_score`: Quality score
- `bid_ask_spread_pct`: Spread percentage
- `regime`: Inferred regime
- `structure_family`: Suggested structure
- `gate_state`: Gate state

## Sample Rich Summary Output

```
╭─────────────────────────────────────────────────────────────────────────╮
│ Summary                                                                  │
╰─────────────────────────────────────────────────────────────────────────╯
ANALYSIS COMPLETE!

The Markov blanket-driven approach has been successfully applied to option pricing.
Key findings:
  • Baseline model MAE: $12.34
  • Classical model MAE: $4.56
  • Combined model MAE: $3.89
  • Improvement: 19.2% (CV MAE reduction)
  • Extra component (V, N) accounts for 0.4% of premium
  • Extra variance explained (residual R²): 0.466 (46.6%)
  • Skew model R²: 0.472 (47.2% of variance)

Kelly Gate:
  • Regime: TRANSFER
  • Suggested structure: CONVEXITY
  • Kelly (raw): 0.0234
  • Kelly (fractional): 0.0059
  • Gate state: DEPLOY
  • Skew: put-call diff=0.0234, slope=0.145, curvature=0.012
  • Term: front=0.18, back=0.16, slope=-0.0002, inverted=True
  • Execution: spread=8.5%, quality=0.72

Results saved to: ../output/analysis_20250115_143022
This demonstrates how Markov blanket features capture market inefficiencies
beyond traditional Black-Scholes assumptions.

╭─────────────────────────────────────────────────────────────────────────╮
│ Kelly Gate                                                              │
╰─────────────────────────────────────────────────────────────────────────╯
Regime: TRANSFER
Structure: CONVEXITY
Kelly (raw): 0.0234
Kelly (fractional): 0.0059
Kelly (adjusted): 0.0042
Gate state: DEPLOY
p: 0.587, b: 2.000
Multiplier: 0.712
```

## Implementation Details

### Regime Inference Logic
- **TRANSFER**: residual_r2 > 0.4 AND skew_r2 > 0.3 (high model confidence + strong skew)
- **PRE_TRANSFER**: residual_r2 > 0.2 OR put_call_iv_diff > 0.1 (moderate inefficiency)
- **PIN**: Otherwise (low inefficiency, efficient pricing)

### Structure Family Selection
- **CONVEXITY**: TRANSFER regime, OR (PRE_TRANSFER + inverted term + steep put skew)
- **MEAN_REVERSION_PREMIUM**: PRE_TRANSFER + normal term + low skew
- **PROBE_ONLY**: PIN regime, OR ambiguous PRE_TRANSFER

### p, b Estimation
- **MEAN_REVERSION_PREMIUM**: p = 0.5 + 0.1 * residual_r2 (max 0.6), b = 1.5
- **CONVEXITY**: p = 0.5 + 0.15 * residual_r2 (max 0.65), b = 2.0
- **PROBE_ONLY**: p = 0.5, b = 1.0
- Shrinkage applied for small sample sizes (< 50)

### Multipliers
- Term inversion: -30% for premium selling, +20% for convexity (capped at 1.5x)
- Steep put skew: -20% for premium selling
- Execution quality: Direct scaling (0-1)
- Regime: PIN = 0.5x, PRE_TRANSFER = 0.8x, TRANSFER = 1.0x

### Gate State
- **BLOCK**: kelly_adjusted <= 0.001 OR quality_score < 0.3
- **PROBE**: kelly_adjusted < 0.01 OR quality_score < 0.5 OR regime == PIN
- **DEPLOY**: Otherwise

## Notes

- All existing outputs and metrics are preserved
- Kelly Gate degrades gracefully if data is missing
- Default values are conservative (safe fallbacks)
- `--debug` flag prints intermediate feature values and p,b estimation details
- Output folder naming convention preserved: `../output/analysis_YYYYMMDD_HHMMSS`

## Teixiptla Anatomy

The Kelly Gate implements the "Teixiptla anatomy" concept:
- **Regime**: Market state (PIN = noise, PRE_TRANSFER = building, TRANSFER = opportunity)
- **Skin selection**: Structure family recommendation based on regime and market structure
- **Mispricing**: Detected via combined model vs classical model difference
- **Kelly permission**: Fractional Kelly with conservative multipliers
- **Transfer logic**: Gate state determines entry permission (BLOCK/PROBE/DEPLOY)

This creates a disciplined permissioning layer that ties mask stability (regime) to market surface structure (skew/term) while preserving existing MAE/R² reporting.
