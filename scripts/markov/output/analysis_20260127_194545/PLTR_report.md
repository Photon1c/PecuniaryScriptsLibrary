# PLTR - Markov Blanket Option Pricing Analysis

**Analysis Date:** 2026-01-27 19:45:51

## Model Performance

- **Baseline MAE:** $1.34
- **Classical Model MAE:** $0.16
- **Combined Model MAE:** $0.10
- **Improvement:** 7.7% (CV MAE reduction)
- **Extra Component (V, N):** 4.4% of premium
- **Extra Variance Explained (Residual R²):** 0.401 (40.1%)
- **Skew Model R²:** 0.213 (21.3%)

## Kelly Gate

- **Regime:** PRE_TRANSFER
- **Structure:** PROBE_ONLY
- **Kelly (raw):** 0.0000
- **Kelly (fractional):** 0.0000
- **Kelly (adjusted):** 0.0000
- **Gate State:** BLOCK
- **p:** 0.500
- **b:** 1.000
- **Multiplier:** 0.588

## State Machine

- **Current State:** COOLDOWN
- **Actions:** Stand down; no new risk while system cools off.
- **Derived from:** regime=PRE_TRANSFER, gate=BLOCK, Kelly=0.0000

## Reflexive Sleeve

**Status:** BLOCKED

Kelly Gate or Teixiptla regime does not permit reflexive nesting.

## Skew Features

- **Put-Call IV Diff:** -0.0000
- **Skew Slope (Puts):** -0.0141
- **Smile Curvature:** 0.0057

## Term Structure

- **Front IV:** 0.0077
- **Back IV:** 0.0073
- **Term Slope:** -0.000086
- **Inverted:** True

## Execution Quality

- **Bid-Ask Spread:** 2.93%
- **Quality Score:** 0.735

