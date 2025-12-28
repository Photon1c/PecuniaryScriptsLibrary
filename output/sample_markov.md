### Sample output for working Markov Mask engine (release in 2026)

# Analysis Summary

## Overall Results

| Metric | Value |
|--------|-------|
| **Baseline Model MAE** | $1.52 |
| **Classical Model MAE** | $0.10 |
| **Combined Model MAE** | $0.08 |
| **Improvement** | 20.1% (CV MAE reduction) |
| **Extra Component (V, N)** | 1.4% of premium |
| **Extra Variance Explained (Residual R²)** | 0.148 (14.8%) |
| **Skew Model R²** | 0.334 (33.4% of variance) |

## Kelly Gate Analysis

| Parameter | Value |
|-----------|-------|
| **Regime** | PIN |
| **Suggested Structure** | PROBE_ONLY |
| **Kelly (raw)** | 0.0000 |
| **Kelly (fractional)** | 0.0000 |
| **Gate State** | BLOCK |
| **p** | 0.500 |
| **b** | 1.000 |
| **Multiplier** | 0.306 |

### Skew Metrics
- Put-call diff: -0.0000
- Slope: -0.0173
- Curvature: 0.0092

### Term Structure
- Front: 0.0014
- Back: 0.0015
- Slope: 0.000002
- Inverted: False

### Execution Quality
- Spread: 4.44%
- Quality: 0.612

## Teixiptla Narrative

ATM contracts show elevated sensitivity without convexity permission. The Markov mask remains intact; agency is present but suppressed. No EXPRESSIVE or RUPTURE_CANDIDATE states detected, indicating the system is operating within PIN constraints.

---

**Results saved to:** `..\output\analysis_20251228_114508`

**Key Insight:** This demonstrates how Markov blanket features capture market inefficiencies beyond traditional Black-Scholes assumptions.
