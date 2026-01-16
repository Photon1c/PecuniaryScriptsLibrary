# Flight Envelope
# option_flight_envelope.py
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import data_loader  # uses your settings.json paths :contentReference[oaicite:1]{index=1}


# ----------------------------
# Stock features (daily)
# ----------------------------
def _to_float(s):
    return pd.to_numeric(s, errors="coerce")

def compute_atr(df: pd.DataFrame, n: int = 14) -> float:
    # With only Close/Last, use close-to-close true range proxy
    close = _to_float(df["Close/Last"]).dropna().values
    if len(close) < n + 2:
        return float(np.nan)
    tr = np.abs(np.diff(close))
    atr = pd.Series(tr).rolling(n).mean().iloc[-1]
    return float(atr)

def compute_hv(df: pd.DataFrame, n: int = 20, annualization: int = 252) -> float:
    close = _to_float(df["Close/Last"]).dropna()
    ret = np.log(close).diff().dropna()
    if len(ret) < n + 2:
        return float(np.nan)
    hv = ret.rolling(n).std().iloc[-1] * np.sqrt(annualization) * 100.0
    return float(hv)

def thrust_proxy(df: pd.DataFrame, n: int = 10) -> float:
    """
    Daily 'CVD-like' proxy:
      thrust ~ sum(sign(return) * volume) over last n days, scaled to [-1,1]
    """
    d = df.copy()
    d["close"] = _to_float(d["Close/Last"])
    d["vol"] = _to_float(d["Volume"]).fillna(0)
    d["ret"] = d["close"].diff()
    d = d.dropna(subset=["ret"]).tail(n)
    if len(d) == 0:
        return 0.0
    signed = np.sign(d["ret"].values) * d["vol"].values
    raw = np.sum(signed)
    scale = np.sum(np.abs(d["vol"].values)) + 1e-9
    return float(raw / scale)  # -1..+1

def available_expirations(chain: pd.DataFrame) -> list[str]:
    if "Expiration Date" not in chain.columns:
        return []
    return sorted(chain["Expiration Date"].dropna().astype(str).str.strip().unique().tolist())


# ----------------------------
# Option structure extraction
# ----------------------------
def normalize_chain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects your quotedata shape: call columns then put columns.
    We only need: Strike, Call/Put OI, Call/Put Gamma, Call/Put IV.
    """
    x = df.copy()
    x["Strike"] = _to_float(x["Strike"])
    # call side
    x["call_oi"] = _to_float(x["Open Interest"])
    x["call_gamma"] = _to_float(x["Gamma"])
    x["call_iv"] = _to_float(x["IV"])
    # put side: these columns share names, so pandas will name them with .1, .2 depending on read
    # Your sample suggests duplicate headers; many CSVs come in as: 'Open Interest.1', 'Gamma.1', 'IV.1'
    for cand in ["Open Interest.1", "Open Interest_1", "Open Interest.2"]:
        if cand in x.columns:
            x["put_oi"] = _to_float(x[cand])
            break
    for cand in ["Gamma.1", "Gamma_1", "Gamma.2"]:
        if cand in x.columns:
            x["put_gamma"] = _to_float(x[cand])
            break
    for cand in ["IV.1", "IV_1", "IV.2"]:
        if cand in x.columns:
            x["put_iv"] = _to_float(x[cand])
            break

    # If any are missing, create them as NaN
    for col in ["put_oi", "put_gamma", "put_iv"]:
        if col not in x.columns:
            x[col] = np.nan

    return x.dropna(subset=["Strike"])

def filter_chain(chain: pd.DataFrame,
                 spot: float,
                 expirations=None,
                 strike_window_pct: float = 0.10):

    c = chain.copy()

    # Normalize Expiration Date strings once
    if "Expiration Date" in c.columns:
        c["Expiration Date"] = c["Expiration Date"].astype(str).str.strip()

    # Normalize user input to list
    if expirations is not None:
        if isinstance(expirations, str):
            expirations = [expirations.strip()]
        else:
            expirations = [str(x).strip() for x in expirations]

        if "Expiration Date" in c.columns:
            c = c[c["Expiration Date"].isin(expirations)]

    # Strike window
    lo = spot * (1.0 - strike_window_pct)
    hi = spot * (1.0 + strike_window_pct)
    c = c[(c["Strike"] >= lo) & (c["Strike"] <= hi)]

    return c

def compute_walls(chain: pd.DataFrame) -> tuple[float, float]:
    if chain is None or len(chain) == 0:
        raise ValueError("Empty option chain after filters; cannot compute walls.")

    g = chain.groupby("Strike", as_index=False)[["call_oi", "put_oi"]].sum()

    if len(g) == 0:
        raise ValueError("No strikes left after grouping; cannot compute walls.")

    call_wall = float(g.loc[g["call_oi"].fillna(0).idxmax(), "Strike"])
    put_wall  = float(g.loc[g["put_oi"].fillna(0).idxmax(), "Strike"])
    return put_wall, call_wall


def compute_flip(chain: pd.DataFrame, spot: float) -> float:
    c = chain.copy()
    c["net_g"] = (c["call_gamma"].fillna(0) * c["call_oi"].fillna(0)) - (c["put_gamma"].fillna(0) * c["put_oi"].fillna(0))

    g = c.groupby("Strike", as_index=False)["net_g"].sum().sort_values("Strike")
    strikes = g["Strike"].values
    net = g["net_g"].values

    s = np.sign(net)
    change = np.where(np.diff(s) != 0)[0]
    if len(change):
        mids = (strikes[change] + strikes[change + 1]) / 2.0
        return float(mids[int(np.argmin(np.abs(mids - spot)))])

    return float(strikes[int(np.argmin(np.abs(net)))])


def current_iv(chain: pd.DataFrame, spot: float) -> float:
    """
    ATM IV proxy: take strikes nearest spot; average call/put IV.
    Your IV values look like percent units already (e.g. 5.10); keep in percent.
    """
    c = chain.copy()
    c["dist"] = np.abs(c["Strike"] - spot)
    atm = c.nsmallest(3, "dist")
    vals = pd.concat([atm["call_iv"], atm["put_iv"]], ignore_index=True).dropna()
    return float(vals.mean()) if len(vals) else float(np.nan)


# ----------------------------
# Envelope computation & plot
# ----------------------------
def build_envelope(ticker: str, risk_budget: float = 1.0, max_loss: float = 1.0, spot_override: float | None = None):
    # 1. Load stock data using the improved loader
    stock = data_loader.load_stock_data(ticker)
    last_row = stock.iloc[-1]
    
    # 2. Derive spot price
    if spot_override is not None:
        spot = float(spot_override)
    else:
        # loader already handles Close/Last vs Close and numeric conversion
        spot = float(last_row["Close/Last"]) if "Close/Last" in last_row else float(last_row["Close"])
    
    print(f"DEBUG: Using spot price: {spot:.2f} (from {'override' if spot_override else 'stock data'})")
    print("STOCK LAST DATE:", last_row["Date"], "LAST CLOSE:", spot)

    # 3. Load option chain data
    chain_raw = data_loader.load_option_chain_data(ticker)
    chain = normalize_chain(chain_raw)

    # pick expirations
    expirations = 'Fri Jan 16 2026'
    print("Available expirations:", available_expirations(chain))

    # 4. Filter chain (one time, no accidental recomputes)
    chain_f = filter_chain(chain, spot=spot, expirations=expirations, strike_window_pct=0.10)

    if len(chain_f) == 0:
        exps = available_expirations(chain)
        raise ValueError(
            "No rows after filtering. Your selected expiration probably doesn't match the CSV.\n"
            f"Selected: {expirations}\n"
            f"Available expirations (copy/paste one): {exps[:15]}{' ...' if len(exps) > 15 else ''}"
        )

    atr = compute_atr(stock, n=14)
    hv = compute_hv(stock, n=20)
    put_wall, call_wall = compute_walls(chain_f)
    flip = compute_flip(chain_f, spot=spot)
    iv = current_iv(chain_f, spot=spot)
    # Axes
    airspeed = abs(spot - flip) / (atr + 1e-9)          # 0..~4+
    thrust = thrust_proxy(stock, n=10)                  # -1..+1
    load = (max_loss / risk_budget)                     # 0..2+

    # Turbulence score (simple v1)
    iv_hv = (iv - hv) if np.isfinite(iv) and np.isfinite(hv) else np.nan
    turbulence = 0.0
    if np.isfinite(iv_hv):
        turbulence = float(np.clip((iv_hv / 15.0), 0, 2))  # ~0 calm, ~1 notable, 2 hot

    # Determine zone
    zone = "NORMAL"
    if airspeed < 0.35:
        zone = "STALL"
    if load > 1.5:
        zone = "STRUCTURAL DAMAGE"
    if airspeed > 3.0 or turbulence > 1.4:
        zone = "TURBULENCE"
    if zone == "NORMAL" and (airspeed > 2.25 or load > 1.0 or turbulence > 1.0):
        zone = "CAUTION"

    # Plot: envelope bands (UAV-style)
    fig = go.Figure()

    # Background zones along airspeed axis
    fig.add_vrect(x0=0, x1=0.35, fillcolor="lightgray", opacity=0.25, line_width=0)
    fig.add_vrect(x0=0.35, x1=2.25, fillcolor="gainsboro", opacity=0.18, line_width=0)
    fig.add_vrect(x0=2.25, x1=3.0, fillcolor="khaki", opacity=0.18, line_width=0)
    fig.add_vrect(x0=3.0, x1=5.0, fillcolor="salmon", opacity=0.16, line_width=0)

    # Structural damage ceiling (load)
    fig.add_hrect(y0=1.5, y1=3.0, fillcolor="salmon", opacity=0.16, line_width=0)

    # Point (your current condition)
    fig.add_trace(go.Scatter(
        x=[airspeed],
        y=[load],
        mode="markers+text",
        text=[f"{ticker}"],
        textposition="top center",
        marker=dict(size=14),
        name="Current"
    ))

    # Secondary annotation: thrust & turbulence
    fig.add_annotation(
        x=airspeed, y=load,
        xref="x", yref="y",
        text=f"Zone: {zone}<br>Airspeed: {airspeed:.2f} ATR<br>Load: {load:.2f}x<br>Thrust: {thrust:+.2f}<br>IV-HV: {iv_hv:+.2f} pp",
        showarrow=True, arrowhead=2, ax=40, ay=-60
    )

    fig.update_layout(
        title=f"Option Flight Envelope — {ticker} | spot={spot:.2f} flip={flip:.2f} walls={put_wall:.0f}/{call_wall:.0f}",
        xaxis_title="Structural Airspeed = |spot − flip| / ATR",
        yaxis_title="Load Factor = MaxLoss / RiskBudget",
        xaxis=dict(range=[0, 5]),
        yaxis=dict(range=[0, 3]),
        height=520
    )

    summary = {
        "ticker": ticker,
        "spot": spot,
        "flip": flip,
        "put_wall": put_wall,
        "call_wall": call_wall,
        "atr": atr,
        "hv": hv,
        "iv_atm": iv,
        "iv_minus_hv_pp": iv_hv,
        "airspeed": airspeed,
        "thrust": thrust,
        "load_factor": load,
        "turbulence_score": turbulence,
        "zone": zone,
    }
    return fig, summary


if __name__ == "__main__":
    # Remove spot_override to test auto-discovery of latest stock data
    fig, summary = build_envelope("SPY")

    print("\n--- Summary Data ---")
    for k, v in summary.items():
        print(f"{k}: {v}")
    
    # Save as HTML to avoid terminal encoding issues and ensure visibility
    output_path = "flight_envelope.html"
    fig.write_html(output_path)
    print(f"\n[SUCCESS] Chart saved to: {output_path}")
    print("Double-click the file to view the Flight Envelope in your browser.")
