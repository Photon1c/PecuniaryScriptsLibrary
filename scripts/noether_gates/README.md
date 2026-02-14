# Noether Early Detection – Regime Gate Smoke Test 🚪💨

## Overview

This project implements a Noether-inspired regime detection overlay on a simple trend-following strategy.

The core research question:

Can we detect regime breaks early — before major drawdowns unfold — and dynamically reduce exposure?

Instead of relying on volatility spikes or price-only triggers, the framework monitors a risk-adjusted log-growth proxy:

$$
Q_t = mu_log − lambda * sigma^2
$$

Where:

- mu_log = rolling mean of log returns
- sigma^2 = rolling variance of log returns
- lambda = risk penalty parameter

Under statistical stationarity (time-translation symmetry), Q_t should remain stable.  
When symmetry breaks, Q_t deteriorates.

This acts as a conservation diagnostic.

---

## Strategy Structure

### Base Strategy

- 20-day momentum  
- Long when Close > SMA(20)

Rolling Kelly fraction sizing:

$$
f_kelly = mu / sigma^2
$$

Smoothed and capped at 1.0 (no leverage)

---

## Regime Gate (Early Detection Engine)

Two detection layers:

1. Negative-side CUSUM on standardized Q_t  
   - Accumulates persistent deterioration  
   - Slowly resets  
   - Detects sustained breakdown  

2. Optional slope confirmation  
   - Penalizes sustained negative drift in Q_t  

Gate function:

$$
f_gated = f_kelly * exp( -kappa * detector )
$$

Exposure contracts as deterioration evidence accumulates.

---

## What the Script Evaluates

The system compares:

- Buy & Hold  
- Pure Rolling Kelly  
- Gated Kelly (Noether overlay)  

Metrics reported:

- CAGR  
- Max Drawdown  
- MAR Ratio  
- Sharpe Ratio  
- Ulcer Index  

---

## Early Detection Evaluation Logic

The script identifies drawdown episodes:

- Episode start: drawdown crosses -5%  
- Event trigger: drawdown reaches -15%  
- Events separated by at least 30 trading days  

For each event, it measures:

- When the gate first reduced exposure  
- Lead time (days before -15% drawdown)  
- Hit rate (percentage of events where the gate reacted early)  

This directly answers:

Does the gate close before major damage?

---

## Why “Noether”?

In physics:

Time-translation symmetry ⇒ conservation of energy.

In this model:

Statistical stationarity ⇒ conservation of risk-adjusted growth.

When that conservation fails, exposure is reduced.

This is not metaphor — it is an operational symmetry-break detector.

---

## File Structure

noether_early_detection.py  
README.md  

---

## Required Input Format

CSV file must contain:

Date, Close/Last

Example path in script:

df = pd.read_csv("F:/inputs/stocks/SPY.csv")

You can swap SPY for QQQ, IWM, etc.

---

## Key Parameters to Tune

Early detection sensitivity:

- CUSUM_FORGET  
- CUSUM_ALPHA  
- KAPPA_CUSUM  
- SLOPE_WIN  
- KAPPA_SLOPE  

More aggressive detection:

- Lower forget rate  
- Shorter slope window  
- Slightly higher kappa  

---

## Suggested Experiments

1. Run on QQQ without retuning.  
2. Increase sensitivity and measure:  
   - Median lead time  
   - Event hit rate  
3. Replace slope layer with:  
   - CUSUM on slope  
   - Rolling t-stat of Q drift  
4. Run block bootstrap (60–120 day blocks) for robustness.  

---

## Expected Behavior

A successful early detector should:

- Reduce Max Drawdown  
- Improve Ulcer Index  
- Maintain similar Sharpe  
- Show positive lead time before most large drawdowns  

If it does not lead, it is reacting — not detecting.

---

## Current Status

This implementation:

- Numerically stable (no NaN / overflow artifacts)  
- Log-return clipped to prevent compounding explosion  
- Produces realistic equity curves  
- Adds measurable drawdown reduction with moderate CAGR tradeoff  
- Demonstrates symmetry-based risk modulation  

---

## Disclaimer

This is research code.

- Not investment advice  
- No transaction costs included  
- Daily data only  
- Rolling windows introduce lag  
- Designed for structural regime research  
