
"""
quantum_tunneling_model.py
--------------------------
Flexible utilities to compute a breakout ("tunneling") probability from an option chain CSV.
Designed to be resilient to varying column names.
"""
from __future__ import annotations
import math
import pandas as pd
from typing import Dict, List, Tuple

# --------- Column mapping helpers ---------
_CANON = {
    "strike": ["strike", "strk", "k"],
    "type": ["type", "option_type", "right", "calls_puts", "cp", "callput"],
    "bid": ["bid", "bid1"],
    "ask": ["ask", "ask1"],
    "last": ["last", "price", "mark"],
    "iv": ["iv", "implied_vol", "impl_vol", "impliedvol"],
    "volume": ["volume", "vol"],
    "open_interest": ["open_interest", "oi", "openint"],
    "gamma": ["gamma", "g"],
    "delta": ["delta", "d"],
    "vega": ["vega", "v"],
    "theta": ["theta", "t"],
    "expiration": ["expiration", "expiry", "exp", "expiration_date", "expiry_date"],
    "underlying": ["underlying", "symbol_root", "ticker", "symbol"],
}

def _first(df: pd.DataFrame, aliases: List[str]) -> str | None:
    for a in aliases:
        if a in df.columns:
            return a
    return None

def map_columns(df: pd.DataFrame) -> Dict[str, str]:
    m = {}
    for k, aliases in _CANON.items():
        got = _first(df, aliases)
        if got: m[k] = got
    return m

def _ensure_mid(row, bid_col, ask_col, last_col) -> float | None:
    b = row.get(bid_col) if bid_col else None
    a = row.get(ask_col) if ask_col else None
    l = row.get(last_col) if last_col else None
    vals = [v for v in [b, a] if isinstance(v, (int, float))]
    if len(vals) == 2 and vals[0] is not None and vals[1] is not None:
        return (vals[0] + vals[1]) / 2.0
    return l if isinstance(l, (int, float)) else None

# --------- Core metrics ---------
def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    m = map_columns(df)

    # normalize option side to 'C'/'P' if we can
    tcol = m.get("type")
    if tcol in df.columns:
        def norm_type(x):
            s = str(x).lower()
            if "c" in s and "p" not in s: return "C"
            if "put" in s or s == "p": return "P"
            if "call" in s or s == "c": return "C"
            # if numeric/other, fallback None
            return None
        df["opt_side"] = df[tcol].map(norm_type)
    else:
        df["opt_side"] = None

    # spread, mid
    bid_col, ask_col, last_col = m.get("bid"), m.get("ask"), m.get("last")
    if bid_col: df["bid"] = pd.to_numeric(df[bid_col], errors="coerce")
    if ask_col: df["ask"] = pd.to_numeric(df[ask_col], errors="coerce")
    if last_col: df["last_trade"] = pd.to_numeric(df[last_col], errors="coerce")
    df["mid"] = df.apply(lambda r: _ensure_mid(r, "bid" if bid_col else None, "ask" if ask_col else None, "last_trade" if last_col else None), axis=1)
    df["spread"] = (df["ask"] - df["bid"]) if bid_col and ask_col else None

    # liquidity depth proxy
    vol_col = m.get("volume"); oi_col = m.get("open_interest")
    if vol_col: df["volume_x"] = pd.to_numeric(df[vol_col], errors="coerce").fillna(0)
    else: df["volume_x"] = 0
    if oi_col: df["oi_x"] = pd.to_numeric(df[oi_col], errors="coerce").fillna(0)
    else: df["oi_x"] = 0

    # barrier height proxy (higher if wide spread + low depth; lower mid prices get scaled)
    # add small epsilons to avoid zero issues
    eps = 1e-9
    df["barrier_height"] = ((df["spread"].fillna(0) + 0.01) * (1.0 / (df["volume_x"] + df["oi_x"] + 1.0)))  # wider spread & thinner book => higher barrier

    # order imbalance proxy at each strike = calls_vol - puts_vol (volume-weighted), needs strike grouping
    kcol = m.get("strike")
    if kcol:
        df["strike_k"] = pd.to_numeric(df[kcol], errors="coerce")
    else:
        df["strike_k"] = None

    # implied vol, gamma if present
    iv_col = m.get("iv")
    if iv_col: df["iv_x"] = pd.to_numeric(df[iv_col], errors="coerce")
    else: df["iv_x"] = None
    g_col = m.get("gamma")
    if g_col: df["gamma_x"] = pd.to_numeric(df[g_col], errors="coerce")
    else: df["gamma_x"] = None

    # compute call/put vol by strike for imbalance
    def side_weighted(group):
        v = group["volume_x"].sum()
        v_c = group.loc[group["opt_side"]=="C", "volume_x"].sum()
        v_p = group.loc[group["opt_side"]=="P", "volume_x"].sum()
        return pd.Series({"vol_total": v, "vol_call": v_c, "vol_put": v_p, "imbalance": v_c - v_p})

    if "strike_k" in df.columns and df["strike_k"].notna().any():
        by_strike = df.groupby("strike_k", dropna=True).apply(side_weighted).reset_index()
    else:
        by_strike = pd.DataFrame(columns=["strike_k","vol_total","vol_call","vol_put","imbalance"])

    df = df.merge(by_strike, on="strike_k", how="left")

    # barrier strength (height scaled by recent IV compression if we have it)
    # If we lack history, approximate compression as inverse of current iv (lower iv => more compression)
    inv_iv = 1.0 / (df["iv_x"].fillna(df["iv_x"].median() or 0.2) + 1e-6)
    df["barrier_strength"] = df["barrier_height"].fillna(0) * inv_iv

    # tunneling probability ~ exp(-k * barrier / (|imbalance| + 1))
    kappa = 500.0  # instead of 3.0
    denom = (df["imbalance"].abs().fillna(0) + 1.0)
    df["tunnel_prob"] = (-(kappa * df["barrier_strength"]) / denom).apply(math.exp)

    # add a qualitative flag
    def flag(p):
        if p >= 0.5: return "HIGH"
        if p >= 0.25: return "MEDIUM"
        return "LOW"
    df["tunnel_flag"] = df["tunnel_prob"].map(flag)

    return df

def focus_window(df: pd.DataFrame, center_strike: float, width: float = 10.0, expiry: str | None = None) -> pd.DataFrame:
    out = df.copy()
    if "strike_k" in out.columns:
        out = out[(out["strike_k"] >= center_strike - width) & (out["strike_k"] <= center_strike + width)]
    if expiry is not None:
        m = map_columns(out)
        e = m.get("expiration")
        if e and e in out.columns:
            out = out[out[e].astype(str) == str(expiry)]
    return out.sort_values(["strike_k","opt_side","mid"], na_position="last")
