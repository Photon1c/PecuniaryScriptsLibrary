# noether_theorem_early_detection.py
# Focus: EARLY DETECTION (A) — does the gate close *before* drawdowns begin?
#
# Strategy: simple 20D momentum + Kelly sizing + Noether-style regime gate.
# Gate detector: fast "negative-side" CUSUM on standardized Q_t (risk-adjusted log-growth proxy)
# + optional short-horizon slope confirmation.
#
# Outputs:
# 1) Strategy metrics (BuyHold / Kelly / Gated)
# 2) Early-detection report: lead-time stats vs drawdown episodes
# 3) Plots: equity, Q, detectors, gate fraction, drawdowns

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# -----------------------------
# User Params
# -----------------------------
CSV_PATH = "F:/inputs/stocks/SPY.csv"   # change to QQQ.csv tomorrow

# Signal / stats
MOM_WINDOW = 20
STATS_WIN  = 252          # rolling window for mu/var and z-scoring Q
KELLY_CAP  = 1.0          # 1.0 = no leverage
EWMA_ALPHA = 0.05         # smooth f_kelly

# Q proxy
LAMBDA_RISK = 1.0         # Q = mu_log - lambda * sigma^2

# Early detector knobs (tuned for early detection)
CUSUM_FORGET = 0.08       # higher = resets faster; lower = more persistent
CUSUM_ALPHA  = 0.25       # smooth CUSUM
KAPPA_CUSUM  = 1.8        # gate strength on CUSUM (typically 0.5–4)

# Optional slope-confirmation (fast-ish)
USE_SLOPE_CONFIRM = True
SLOPE_WIN    = 25         # shorter = earlier but noisier
SLOPE_ALPHA  = 0.30       # smooth slope z
KAPPA_SLOPE  = 0.8        # small extra penalty
SLOPE_FORGET = 0.02       # small forget term (optional)

# Exposure behavior
FLOOR_ON_SIGNAL = 0.0     # set 0.10–0.25 if you want "never fully off" when Signal=1
MAX_DAILY_STRAT_LOG = 0.12  # safety cap for strategy daily log return (not necessary but stable)

# Early-detection evaluation
DD_START_THRESHOLD = 0.05  # start of an episode = drawdown crosses -5%
DD_EVENT_THRESHOLD = 0.15  # event = drawdown reaches -15%
MIN_GAP_BARS = 30          # separate events by at least N bars
GATE_CLOSE_Q = 0.25        # "gate closed" if gate_factor <= this (0.25 = heavy constriction)

# -----------------------------
# Helpers
# -----------------------------
def clean_price(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["Date", "Close/Last"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = (
        df["Close/Last"].astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .astype(float)
    )
    df = df.sort_values("Date").drop_duplicates(subset="Date", keep="last").set_index("Date")
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1)).clip(-0.20, 0.20)
    df = df.dropna(subset=["LogReturn"])
    return df

def rolling_mu_var(logret: pd.Series, win: int):
    mu = logret.rolling(win).mean()
    var = logret.rolling(win).var().clip(lower=1e-8)
    return mu, var

def compute_slope(series: pd.Series, win: int) -> pd.Series:
    out = np.full(len(series), np.nan)
    vals = series.values
    for i in range(win - 1, len(series)):
        y = vals[i - win + 1:i + 1]
        if np.isnan(y).any():
            continue
        x = np.arange(win)
        slope, _, _, _, _ = linregress(x, y)
        out[i] = slope
    return pd.Series(out, index=series.index)

def cusum_negative(z: pd.Series, forget: float) -> pd.Series:
    """
    Negative-side CUSUM on z:
    - z < 0 indicates Q below its rolling mean (bad)
    - accumulate -z when negative, subtract 'forget' each step, floor at 0
    """
    neg = (-z).clip(lower=0).fillna(0)
    c = np.zeros(len(neg), dtype=float)
    nv = neg.values
    for i in range(1, len(nv)):
        c[i] = max(0.0, c[i - 1] + nv[i] - forget)
    return pd.Series(c, index=z.index)

def eq_from_logret(logret: pd.Series) -> pd.Series:
    return np.exp(logret.fillna(0).cumsum())

def max_drawdown(eq: pd.Series) -> float:
    dd = (eq / eq.cummax() - 1.0)
    return float(dd.min())

def metrics(eq: pd.Series, logret: pd.Series) -> dict:
    simple = (np.exp(logret.dropna()) - 1.0)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan
    dd = max_drawdown(eq)
    ulcer = float(np.sqrt(((eq / eq.cummax() - 1.0) ** 2).mean()))
    sharpe = float(simple.mean() / simple.std() * np.sqrt(252)) if simple.std() != 0 else np.nan
    mar = float(cagr / (-dd)) if dd < 0 else np.nan
    return {
        "CAGR (%)": round(100 * cagr, 2),
        "Max DD (%)": round(100 * dd, 2),
        "MAR": round(mar, 2),
        "Sharpe": round(sharpe, 2),
        "Ulcer (%)": round(100 * ulcer, 2),
    }

def drawdown_series(eq: pd.Series) -> pd.Series:
    return (eq / eq.cummax() - 1.0)

def find_dd_events(dd: pd.Series, start_th: float, event_th: float, min_gap: int):
    """
    Identify drawdown episodes:
      - start when dd <= -start_th
      - event when dd <= -event_th
    Return list of dicts with start_date, event_date, end_date (recovery to above -start_th).
    """
    ddv = dd.values
    idx = dd.index
    events = []
    in_ep = False
    start_i = None
    event_i = None

    last_event_i = -10**9

    for i in range(len(ddv)):
        if not in_ep:
            if ddv[i] <= -start_th and (i - last_event_i) >= min_gap:
                in_ep = True
                start_i = i
                event_i = None
        else:
            if event_i is None and ddv[i] <= -event_th:
                event_i = i
            # end when we recover above -start_th
            if ddv[i] > -start_th:
                end_i = i
                # only keep if it reached event threshold
                if event_i is not None:
                    events.append({
                        "start_date": idx[start_i],
                        "event_date": idx[event_i],
                        "end_date": idx[end_i],
                        "start_i": start_i,
                        "event_i": event_i,
                        "end_i": end_i,
                        "min_dd": float(dd.iloc[start_i:end_i+1].min()),
                    })
                    last_event_i = event_i
                in_ep = False
                start_i = None
                event_i = None

    return events

def lead_time_report(df: pd.DataFrame, gate_factor: pd.Series, eq_ref: pd.Series):
    """
    For each drawdown event in eq_ref, measure earliest 'gate close' prior to event_date.
    gate close condition: gate_factor <= GATE_CLOSE_Q
    Lead = (event_date - first_close_date).days, if first_close_date < event_date.
    """
    dd = drawdown_series(eq_ref)
    events = find_dd_events(dd, DD_START_THRESHOLD, DD_EVENT_THRESHOLD, MIN_GAP_BARS)
    closes = (gate_factor <= GATE_CLOSE_Q).fillna(False)

    rows = []
    for e in events:
        s, ev, en = e["start_date"], e["event_date"], e["end_date"]
        window = closes.loc[s:ev]
        if window.any():
            first_close = window[window].index[0]
            lead_days = (ev - first_close).days
        else:
            first_close = pd.NaT
            lead_days = None

        rows.append({
            "start": s.date(),
            "event": ev.date(),
            "min_dd_%": round(100 * e["min_dd"], 2),
            "first_gate_close": None if pd.isna(first_close) else first_close.date(),
            "lead_days": lead_days,
        })

    rep = pd.DataFrame(rows)
    if len(rep) == 0:
        return rep, {"events": 0}

    lead_vals = rep["lead_days"].dropna().astype(float)
    summary = {
        "events": int(len(rep)),
        "events_with_gate_close": int(lead_vals.shape[0]),
        "close_hit_rate": float(lead_vals.shape[0] / len(rep)),
        "lead_days_median": float(lead_vals.median()) if lead_vals.shape[0] else np.nan,
        "lead_days_mean": float(lead_vals.mean()) if lead_vals.shape[0] else np.nan,
        "lead_days_min": float(lead_vals.min()) if lead_vals.shape[0] else np.nan,
        "lead_days_max": float(lead_vals.max()) if lead_vals.shape[0] else np.nan,
    }
    return rep, summary

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(CSV_PATH)
df = clean_price(df)
print(f"Data range: {df.index.min()} → {df.index.max()} | {len(df)} bars")

# -----------------------------
# Signal
# -----------------------------
df["SMA"] = df["Close"].rolling(MOM_WINDOW).mean()
df["Signal"] = (df["Close"] > df["SMA"]).astype(float)

# -----------------------------
# Q_t and Kelly sizing (base)
# -----------------------------
mu, var = rolling_mu_var(df["LogReturn"], STATS_WIN)
f_kelly_raw = (mu / var).clip(0, 2.0)
f_kelly = f_kelly_raw.ewm(alpha=EWMA_ALPHA, adjust=False).mean().clip(0, KELLY_CAP)

Q_t = (mu - LAMBDA_RISK * var)

# Standardize Q for early detection (fast break detection wants z-scores)
Q_mu = Q_t.rolling(STATS_WIN).mean()
Q_sd = Q_t.rolling(STATS_WIN).std().clip(lower=1e-6)
Q_z = ((Q_t - Q_mu) / Q_sd).fillna(0)

# -----------------------------
# Early detector 1: CUSUM on negative Q_z
# -----------------------------
cus = cusum_negative(Q_z, CUSUM_FORGET).ewm(alpha=CUSUM_ALPHA, adjust=False).mean().fillna(0)

gate_factor = np.exp(-KAPPA_CUSUM * cus)

# -----------------------------
# Optional early detector 2: slope confirmation (penalize negative slope z)
# -----------------------------
if USE_SLOPE_CONFIRM:
    Q_slope = compute_slope(Q_t, SLOPE_WIN).fillna(0)
    sl_sd = Q_slope.rolling(STATS_WIN).std().clip(lower=1e-8)
    sl_z = (Q_slope / sl_sd).ewm(alpha=SLOPE_ALPHA, adjust=False).mean().fillna(0)

    # only negative slope adds penalty
    sl_pen = (-sl_z).clip(lower=0)

    # optional "forget" (keeps slope penalty from sticking forever)
    sl_cus = cusum_negative(-sl_z, SLOPE_FORGET).ewm(alpha=0.35, adjust=False).mean().fillna(0)

    # combine: cusum-based + slope-based
    gate_factor = gate_factor * np.exp(-KAPPA_SLOPE * sl_pen) * np.exp(-0.6 * sl_cus)
    gate_factor = gate_factor.clip(0, 1)

# -----------------------------
# Gated exposure
# -----------------------------
f_gated = (f_kelly * gate_factor).clip(0, KELLY_CAP)

# conditional floor only when signal=1
if FLOOR_ON_SIGNAL > 0:
    f_gated = pd.Series(
        np.where(df["Signal"] > 0, np.maximum(f_gated, FLOOR_ON_SIGNAL * f_kelly), 0.0),
        index=df.index
    )
else:
    f_gated = pd.Series(np.where(df["Signal"] > 0, f_gated, 0.0), index=df.index)

# -----------------------------
# Strategy returns (log)
# -----------------------------
strat_k = (df["Signal"].shift(1) * df["LogReturn"] * f_kelly.shift(1)).fillna(0)
strat_g = (df["Signal"].shift(1) * df["LogReturn"] * f_gated.shift(1)).fillna(0)

strat_k = strat_k.clip(-MAX_DAILY_STRAT_LOG, MAX_DAILY_STRAT_LOG)
strat_g = strat_g.clip(-MAX_DAILY_STRAT_LOG, MAX_DAILY_STRAT_LOG)

eq_bh = eq_from_logret(df["LogReturn"]).rename("BuyHold")
eq_k  = eq_from_logret(strat_k).rename("Kelly")
eq_g  = eq_from_logret(strat_g).rename("Gated")

# -----------------------------
# Print metrics
# -----------------------------
print("Buy & Hold:", metrics(eq_bh, df["LogReturn"]))
print("Pure Kelly:", metrics(eq_k, strat_k))
print("Gated Kelly:", metrics(eq_g, strat_g))

# -----------------------------
# Early detection report
# Use Buy&Hold drawdowns as the reference "market drawdown episodes"
# (You can also evaluate vs eq_k drawdowns if you want "strategy-protection lead time".)
# -----------------------------
rep, summary = lead_time_report(df, gate_factor, eq_bh)
print("\n--- Early Detection Report (reference: Buy&Hold drawdowns) ---")
print(rep.to_string(index=False) if len(rep) else "No events found with given thresholds.")
print("\n--- Early Detection Summary ---")
for k, v in summary.items():
    print(f"{k}: {v}")

# -----------------------------
# Plots
# -----------------------------
dd_bh = drawdown_series(eq_bh)
dd_k  = drawdown_series(eq_k)
dd_g  = drawdown_series(eq_g)

fig, axs = plt.subplots(5, 1, figsize=(14, 14), sharex=True)

# 1) Equity
eq_bh.plot(ax=axs[0], label="BuyHold")
eq_k.plot(ax=axs[0], label="Kelly")
eq_g.plot(ax=axs[0], label="Gated")
axs[0].legend()
axs[0].set_title("Equity Curves")

# 2) Drawdowns
dd_bh.plot(ax=axs[1], label="DD BuyHold")
dd_k.plot(ax=axs[1], label="DD Kelly", alpha=0.7)
dd_g.plot(ax=axs[1], label="DD Gated", alpha=0.7)
axs[1].axhline(-DD_START_THRESHOLD, linestyle="--")
axs[1].axhline(-DD_EVENT_THRESHOLD, linestyle="--")
axs[1].legend()
axs[1].set_title("Drawdowns (episode thresholds)")

# 3) Q and Q_z
Q_t.plot(ax=axs[2], label="Q_t")
Q_z.plot(ax=axs[2], label="Q_z", alpha=0.6)
axs[2].legend()
axs[2].set_title("Q_t and standardized Q_z")

# 4) Detectors
cus.plot(ax=axs[3], label="CUSUM(Q_z-) (smoothed)")
axs[3].legend()
axs[3].set_title("Early detector (CUSUM)")

# 5) Gate factor and exposures
gate_factor.plot(ax=axs[4], label="gate_factor", color="C1")
f_kelly.plot(ax=axs[4], label="f_kelly", alpha=0.5)
f_gated.plot(ax=axs[4], label="f_gated", alpha=0.8)
axs[4].axhline(GATE_CLOSE_Q, linestyle="--", color="C1", alpha=0.6)
axs[4].legend()
axs[4].set_title("Gate factor and allocations")

plt.tight_layout()
plt.show()
