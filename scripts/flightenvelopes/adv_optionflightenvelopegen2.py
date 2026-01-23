"""
Option Flight Envelope Generator 2

Generates a chart mapping stock tickers' load factor in relation to their structural airspeed

"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Optional: adjustText for label positioning
try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False

import data_loader  # <- your attached module :contentReference[oaicite:2]{index=2}

# -----------------------------
# Config (can be overridden via CLI args)
# -----------------------------
DEALER_PRESSURE_PATH: Optional[Path] = None
RRE_TOP_N: int = 8
RRE_ZWIN: int = 60
ATR_N: int = 14
HIGHLIGHT_TICKERS: List[str] = []  # e.g., ["AMD"]


@dataclass
class WallResult:
    put_wall: Optional[float]
    call_wall: Optional[float]
    band_pct_used: Optional[float]
    method: str


def _get_close_series(stock_df: pd.DataFrame) -> pd.Series:
    if "Close/Last" in stock_df.columns:
        s = stock_df["Close/Last"]
    elif "Close" in stock_df.columns:
        s = stock_df["Close"]
    else:
        raise ValueError(f"Stock DF missing Close column. Columns: {stock_df.columns.tolist()}")
    return pd.to_numeric(s, errors="coerce").dropna()


def compute_hv20(stock_df: pd.DataFrame) -> float:
    """Compute 20-day historical volatility, annualized, as percentage."""
    close = _get_close_series(stock_df)
    rets = close.pct_change().dropna()
    if len(rets) < 25:
        return float("nan")
    # Annualize and convert to percentage (multiply by 100)
    hv_ann_pct = float(rets.tail(20).std(ddof=1) * math.sqrt(252) * 100)
    return hv_ann_pct


def compute_atr14_close_to_close(stock_df: pd.DataFrame) -> float:
    # If you later have High/Low, swap this for true ATR.
    close = _get_close_series(stock_df)
    tr = close.diff().abs().dropna()
    if len(tr) < 20:
        return float("nan")
    return float(tr.tail(14).mean())


# -----------------------------
# NEW: RRE (Realized Range Expansion) helpers
# -----------------------------
def compute_true_range(df: pd.DataFrame) -> pd.Series:
    """Compute True Range from OHLCV data. Falls back to abs(close.diff()) if High/Low missing."""
    close = _get_close_series(df)
    
    # Try to get High and Low
    high = None
    low = None
    if "High" in df.columns:
        high = pd.to_numeric(df["High"], errors="coerce")
    elif "High/Last" in df.columns:
        high = pd.to_numeric(df["High/Last"], errors="coerce")
    
    if "Low" in df.columns:
        low = pd.to_numeric(df["Low"], errors="coerce")
    elif "Low/Last" in df.columns:
        low = pd.to_numeric(df["Low/Last"], errors="coerce")
    
    if high is not None and low is not None and high.notna().any() and low.notna().any():
        # True Range: max(High-Low, abs(High-prevClose), abs(Low-prevClose))
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    else:
        # Fallback: use abs(close.diff())
        tr = close.diff().abs()
    
    return tr


def compute_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Compute ATR (Average True Range) with window n."""
    tr = compute_true_range(df)
    if len(tr) < n:
        return pd.Series(dtype=float)
    return tr.rolling(window=n, min_periods=1).mean()


def compute_rre_metrics(df: pd.DataFrame, atr_n: int = 14, zwin: int = 60) -> Tuple[float, float]:
    """
    Compute RRE (Realized Range Expansion) metrics.
    Returns: (rre_raw_latest, rre_z_latest)
    """
    tr = compute_true_range(df)
    atr = compute_atr(df, n=atr_n)
    
    if len(tr) < atr_n or len(atr) < atr_n:
        return float(np.nan), float(np.nan)
    
    # RRE_raw: TR_today / ATR_14
    rre_raw = tr / (atr + 1e-9)
    
    if len(rre_raw) < zwin:
        # Not enough data for z-score
        rre_raw_latest = float(rre_raw.iloc[-1]) if len(rre_raw) > 0 and np.isfinite(rre_raw.iloc[-1]) else float(np.nan)
        return rre_raw_latest, float(np.nan)
    
    # Compute z-score over rolling window
    rolling_mean = rre_raw.rolling(window=zwin, min_periods=zwin).mean()
    rolling_std = rre_raw.rolling(window=zwin, min_periods=zwin).std(ddof=0)
    
    rre_z = (rre_raw - rolling_mean) / (rolling_std + 1e-9)
    
    rre_raw_latest = float(rre_raw.iloc[-1]) if np.isfinite(rre_raw.iloc[-1]) else float(np.nan)
    rre_z_latest = float(rre_z.iloc[-1]) if np.isfinite(rre_z.iloc[-1]) else float(np.nan)
    
    return rre_raw_latest, rre_z_latest


# -----------------------------
# NEW: Dealer Pressure Sign helpers
# -----------------------------
def load_dealer_pressure(path: Optional[Path]) -> Dict[str, float]:
    """Load dealer pressure data from JSON or CSV file. Returns dict[ticker] -> net_gamma_at_spot."""
    if path is None or not path.exists():
        return {}
    
    result = {}
    try:
        if path.suffix.lower() == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Handle both {ticker: value} and {ticker: {net_gamma_at_spot: value}} formats
                for ticker, val in data.items():
                    if isinstance(val, dict):
                        result[ticker.upper()] = float(val.get('net_gamma_at_spot', 0))
                    else:
                        result[ticker.upper()] = float(val)
        else:
            # Assume CSV with columns: ticker, net_gamma_at_spot
            df = pd.read_csv(path)
            if 'ticker' in df.columns and 'net_gamma_at_spot' in df.columns:
                for _, row in df.iterrows():
                    result[row['ticker'].upper()] = float(row['net_gamma_at_spot'])
    except Exception as e:
        print(f"WARNING: Failed to load dealer pressure from {path}: {e}")
    
    return result


def dealer_pressure_sign(net_gamma_at_spot: Optional[float]) -> str:
    """Return dealer pressure sign: '+' if > 0, '-' if < 0, '?' if missing."""
    if net_gamma_at_spot is None or not np.isfinite(net_gamma_at_spot):
        return "?"
    if net_gamma_at_spot > 0:
        return "+"
    if net_gamma_at_spot < 0:
        return "-"
    return "?"


def _coerce_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)


def find_near_spot_walls(
    chain_df: pd.DataFrame,
    spot: float,
    strike_col: str = "Strike",
    call_oi_col: str = "Open Interest",
    put_oi_col: str = "Open Interest.1",
    # Many chain dumps duplicate column names; if yours are different,
    # update these two columns once and keep everything else unchanged.
    band_pcts: Tuple[float, ...] = (0.03, 0.05, 0.08, 0.12),
) -> WallResult:
    if strike_col not in chain_df.columns:
        return WallResult(None, None, None, "missing_strike_col")

    strikes = pd.to_numeric(chain_df[strike_col], errors="coerce")
    calls_oi = _coerce_numeric(chain_df, call_oi_col)
    puts_oi = _coerce_numeric(chain_df, put_oi_col)

    df = chain_df.copy()
    df["_strike"] = strikes
    df["_call_oi"] = calls_oi
    df["_put_oi"] = puts_oi
    df = df.dropna(subset=["_strike"])

    # Guard: if OI cols are missing or all NaN, fail gracefully
    if df["_call_oi"].notna().sum() == 0 and df["_put_oi"].notna().sum() == 0:
        return WallResult(None, None, None, "missing_or_empty_OI_cols")

    for band_pct in band_pcts:
        lo, hi = spot * (1 - band_pct), spot * (1 + band_pct)
        w = df[(df["_strike"] >= lo) & (df["_strike"] <= hi)].copy()
        if len(w) == 0:
            continue

        call_wall = None
        put_wall = None

        if w["_call_oi"].notna().any():
            i = w["_call_oi"].fillna(-1).idxmax()
            call_wall = float(w.loc[i, "_strike"]) if i in w.index else None

        if w["_put_oi"].notna().any():
            i = w["_put_oi"].fillna(-1).idxmax()
            put_wall = float(w.loc[i, "_strike"]) if i in w.index else None

        if call_wall is not None or put_wall is not None:
            return WallResult(put_wall, call_wall, float(band_pct), "max_OI_within_dynamic_band")

    return WallResult(None, None, None, "no_walls_found")


def expected_move_from_hv(spot: float, hv_ann: float, days: int) -> float:
    if not np.isfinite(hv_ann):
        return float("nan")
    return float(spot * hv_ann * math.sqrt(days / 252))


def _to_float(s):
    """Helper to convert to float (matches flight_envelope.py)."""
    return pd.to_numeric(s, errors="coerce")


def _normalize_chain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize option chain columns (matches flight_envelope.py exactly).
    Expects your quotedata shape: call columns then put columns.
    We only need: Strike, Call/Put OI, Call/Put Gamma, Call/Put IV.
    """
    x = df.copy()
    x["Strike"] = _to_float(x["Strike"])
    
    # Call side - use direct column access like flight_envelope.py
    x["call_oi"] = _to_float(x["Open Interest"])
    x["call_gamma"] = _to_float(x["Gamma"])
    x["call_iv"] = _to_float(x["IV"])
    
    # Put side - check for duplicate column names (pandas adds .1, .2, etc.)
    # Many chain dumps duplicate column names; pandas will name them with .1, .2 depending on read
    for cand in ["Open Interest.1", "Open Interest_1", "Open Interest.2"]:
        if cand in x.columns:
            x["put_oi"] = _to_float(x[cand])
            break
    else:
        x["put_oi"] = pd.Series(dtype=float)
    
    for cand in ["Gamma.1", "Gamma_1", "Gamma.2"]:
        if cand in x.columns:
            x["put_gamma"] = _to_float(x[cand])
            break
    else:
        x["put_gamma"] = pd.Series(dtype=float)
    
    for cand in ["IV.1", "IV_1", "IV.2"]:
        if cand in x.columns:
            x["put_iv"] = _to_float(x[cand])
            break
    else:
        x["put_iv"] = pd.Series(dtype=float)
    
    # If any are missing, create them as NaN (like flight_envelope.py)
    for col in ["put_oi", "put_gamma", "put_iv"]:
        if col not in x.columns:
            x[col] = np.nan
    
    return x.dropna(subset=["Strike"])


def _compute_net_gamma_at_spot(chain: pd.DataFrame, spot: float) -> float:
    """
    Compute net gamma exposure at spot price from option chain.
    Net gamma = sum of (call_gamma * call_oi) - (put_gamma * put_oi) across all strikes.
    This represents dealer's net gamma position which affects price stability.
    """
    c = chain.copy()
    c["net_g"] = (c.get("call_gamma", pd.Series()).fillna(0) * c.get("call_oi", pd.Series()).fillna(0)) - \
                 (c.get("put_gamma", pd.Series()).fillna(0) * c.get("put_oi", pd.Series()).fillna(0))
    
    # Sum net gamma across all strikes
    net_gamma_total = float(c["net_g"].sum())
    
    return net_gamma_total


def _compute_flip(chain: pd.DataFrame, spot: float) -> float:
    """Compute flip point from option chain (gamma flip)."""
    c = chain.copy()
    c["net_g"] = (c.get("call_gamma", pd.Series()).fillna(0) * c.get("call_oi", pd.Series()).fillna(0)) - \
                 (c.get("put_gamma", pd.Series()).fillna(0) * c.get("put_oi", pd.Series()).fillna(0))
    
    g = c.groupby("Strike", as_index=False)["net_g"].sum().sort_values("Strike")
    strikes = g["Strike"].values
    net = g["net_g"].values
    
    s = np.sign(net)
    change = np.where(np.diff(s) != 0)[0]
    if len(change):
        mids = (strikes[change] + strikes[change + 1]) / 2.0
        return float(mids[int(np.argmin(np.abs(mids - spot)))])
    
    return float(strikes[int(np.argmin(np.abs(net)))])


def _current_iv(chain: pd.DataFrame, spot: float) -> float:
    """
    ATM IV proxy: take strikes nearest spot; average call/put IV.
    IV values in CSV are stored as decimals (e.g. 0.1369 = 13.69%); convert to percentage.
    """
    c = chain.copy()
    c["dist"] = np.abs(c["Strike"] - spot)
    atm = c.nsmallest(3, "dist")
    
    # Get call_iv and put_iv values (they should exist after normalization)
    call_ivs = atm["call_iv"].dropna() if "call_iv" in atm.columns else pd.Series(dtype=float)
    put_ivs = atm["put_iv"].dropna() if "put_iv" in atm.columns else pd.Series(dtype=float)
    
    # Combine and average
    vals = pd.concat([call_ivs, put_ivs], ignore_index=True).dropna()
    
    if len(vals) == 0:
        print(f"WARNING: No IV values found for spot={spot:.2f}")
        return float(np.nan)
    
    # IV values are in decimal format (0.1369), convert to percentage (13.69)
    iv_mean_decimal = float(vals.mean())
    iv_mean_percent = iv_mean_decimal * 100.0
    
    print(f"DEBUG IV: spot={spot:.2f}, found {len(vals)} IV values")
    print(f"  Raw decimal mean: {iv_mean_decimal:.4f}, converted to percent: {iv_mean_percent:.2f}%")
    print(f"  call_iv samples (decimal): {call_ivs.head(3).tolist() if len(call_ivs) > 0 else 'none'}")
    print(f"  put_iv samples (decimal): {put_ivs.head(3).tolist() if len(put_ivs) > 0 else 'none'}")
    
    return iv_mean_percent


def compute_envelope_for_ticker(
    ticker: str,
    out_dir: Path,
    make_charts: bool = False,
) -> Dict:
    stock_df = data_loader.load_stock_data(ticker)  # :contentReference[oaicite:3]{index=3}
    spot = float(data_loader.get_latest_price(ticker))  # :contentReference[oaicite:4]{index=4}

    hv20 = compute_hv20(stock_df)
    atr14 = compute_atr14_close_to_close(stock_df)
    
    # Compute RRE metrics
    rre_raw, rre_z = compute_rre_metrics(stock_df, atr_n=ATR_N, zwin=RRE_ZWIN)

    chain_date = data_loader.get_most_recent_option_date(ticker)  # :contentReference[oaicite:5]{index=5}
    chain_df = data_loader.load_option_chain_data(ticker)  # :contentReference[oaicite:6]{index=6}

    walls = find_near_spot_walls(chain_df, spot)
    
    # Debug: check raw chain columns before normalization
    print(f"DEBUG {ticker}: Raw chain columns: {chain_df.columns.tolist()[:10]}...")
    if "IV" in chain_df.columns:
        sample_raw_iv = chain_df["IV"].dropna().head(3)
        print(f"DEBUG {ticker}: Raw IV column sample: {sample_raw_iv.tolist()}")
    
    # Normalize chain and compute flip/IV/net_gamma_at_spot
    chain_norm = _normalize_chain(chain_df)
    
    if len(chain_norm) == 0:
        print(f"WARNING: Empty chain after normalization for {ticker}")
        flip = spot
        iv = float(np.nan)
        net_gamma_at_spot = float(np.nan)
    else:
        # Debug: print column names to verify IV columns exist
        print(f"DEBUG {ticker}: Normalized chain columns: {[c for c in chain_norm.columns if 'iv' in c.lower()]}")
        if "call_iv" in chain_norm.columns:
            sample_iv = chain_norm["call_iv"].dropna()
            if len(sample_iv) > 0:
                print(f"DEBUG {ticker}: Normalized call_iv sample: {sample_iv.head(5).tolist()}")
                print(f"DEBUG {ticker}: call_iv stats: min={sample_iv.min():.2f}, max={sample_iv.max():.2f}, mean={sample_iv.mean():.2f}")
        
        flip = _compute_flip(chain_norm, spot)
        iv = _current_iv(chain_norm, spot)
        net_gamma_at_spot = _compute_net_gamma_at_spot(chain_norm, spot)
    
    # Compute flight envelope coordinates
    # X: Structural Airspeed = |spot - flip| / ATR
    structural_airspeed = abs(spot - flip) / (atr14 + 1e-9) if np.isfinite(atr14) and atr14 > 0 else float(np.nan)
    # Y: Load Factor = IV / HV (both should be in percent already)
    load_factor = iv / (hv20 + 1e-9) if np.isfinite(iv) and np.isfinite(hv20) and hv20 > 0 else float(np.nan)

    # Compute wall distances in ATR space (for multi-ticker charts)
    x_call_wall_atr = None
    x_put_wall_atr = None
    x_nearest_wall_atr = None
    
    if walls.call_wall is not None and np.isfinite(spot) and np.isfinite(atr14) and atr14 > 0:
        x_call_wall_atr = abs(walls.call_wall - spot) / (atr14 + 1e-9)
    
    if walls.put_wall is not None and np.isfinite(spot) and np.isfinite(atr14) and atr14 > 0:
        x_put_wall_atr = abs(spot - walls.put_wall) / (atr14 + 1e-9)
    
    if x_call_wall_atr is not None and x_put_wall_atr is not None:
        x_nearest_wall_atr = min(x_call_wall_atr, x_put_wall_atr)
    elif x_call_wall_atr is not None:
        x_nearest_wall_atr = x_call_wall_atr
    elif x_put_wall_atr is not None:
        x_nearest_wall_atr = x_put_wall_atr

    # expected_move_from_hv expects HV as decimal (not percentage)
    hv20_decimal = hv20 / 100.0 if np.isfinite(hv20) else float(np.nan)
    em_5 = expected_move_from_hv(spot, hv20_decimal, 5)
    em_10 = expected_move_from_hv(spot, hv20_decimal, 10)
    em_20 = expected_move_from_hv(spot, hv20_decimal, 20)

    result = {
        "ticker": ticker.upper(),
        "as_of_stock_date": str(stock_df["Date"].iloc[-1]) if "Date" in stock_df.columns and len(stock_df) else None,
        "as_of_chain_date": chain_date,
        "spot": spot,
        "flip": flip,
        "iv": iv,
        "hv20": hv20,
        "atr14_close_to_close": atr14,
        "structural_airspeed": structural_airspeed,
        "load_factor": load_factor,
        "expected_move": {"5d": em_5, "10d": em_10, "20d": em_20},
        "envelope": {
            "hv_band_10d": [spot - em_10, spot + em_10] if np.isfinite(em_10) else [None, None],
            "atr_band_1x": [spot - atr14, spot + atr14] if np.isfinite(atr14) else [None, None],
            "atr_band_2x": [spot - 2 * atr14, spot + 2 * atr14] if np.isfinite(atr14) else [None, None],
        },
        "walls": {
            "put_wall": walls.put_wall,
            "call_wall": walls.call_wall,
            "band_pct_used": walls.band_pct_used,
            "method": walls.method,
        },
        "wall_distances_atr": {
            "x_call_wall_atr": x_call_wall_atr,
            "x_put_wall_atr": x_put_wall_atr,
            "x_nearest_wall_atr": x_nearest_wall_atr,
        },
        "rre": {
            "rre_raw": rre_raw,
            "rre_z": rre_z,
        },
        "net_gamma_at_spot": net_gamma_at_spot,
    }

    return result


def _save_detailed_csv(raw_points: List[Dict], csv_path: Path) -> None:
    """Save detailed metrics to CSV file."""
    rows = []
    for rp in raw_points:
        row = {
            "ticker": rp["ticker"],
            "spot": rp.get("spot"),
            "flip": rp.get("flip"),
            "structural_airspeed": rp.get("structural_airspeed") or rp.get("x"),
            "load_factor": rp.get("load_factor") or rp.get("y_raw"),
            "iv": rp.get("iv"),
            "hv20": rp.get("hv"),
            "atr14": rp.get("atr"),
            "put_wall": rp.get("put_wall"),
            "call_wall": rp.get("call_wall"),
            "wall_distance_atr": rp.get("x_nearest_wall_atr"),
            "rre_raw": rp.get("rre_raw"),
            "rre_z": rp.get("rre_z"),
            "net_gamma_at_spot": rp.get("net_gamma_at_spot"),
            "dps": rp.get("dps"),
            "expected_move_5d": rp.get("expected_move_5d"),
            "expected_move_10d": rp.get("expected_move_10d"),
            "expected_move_20d": rp.get("expected_move_20d"),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved {len(rows)} tickers to detailed CSV")


def plot_flight_envelope_snapshot(
    all_results: Dict[str, Dict],
    output_path: Path,
    dealer_pressure_data: Optional[Dict[str, float]] = None,
    rre_top_n: int = 8,
    highlight_tickers: Optional[List[str]] = None,
) -> None:
    """Create a single scatter plot with all tickers on the flight envelope chart."""
    if highlight_tickers is None:
        highlight_tickers = []
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Collect valid data points (with clamped/log load factor for display)
    points = []
    raw_points = []  # Keep raw values for annotations
    skipped_tickers = []
    
    # Collect RRE z-scores to determine top N
    rre_z_scores = []
    
    for ticker, data in all_results.items():
        if "error" in data:
            skipped_tickers.append(f"{ticker} (error: {data.get('error', 'unknown')})")
            continue
        
        x = data.get("structural_airspeed")
        y_raw = data.get("load_factor")
        
        # Debug: check why points might be skipped
        if not np.isfinite(x):
            skipped_tickers.append(f"{ticker} (invalid x={x})")
            continue
        if not np.isfinite(y_raw):
            skipped_tickers.append(f"{ticker} (invalid y={y_raw})")
            continue
        
        # Clamp load factor for visibility (while keeping raw for display)
        y_display = min(y_raw, 5.0)  # Clamp at 5.0 for visibility
        wall_dist = data.get("wall_distances_atr", {})
        rre_data = data.get("rre", {})
        rre_z = rre_data.get("rre_z", float(np.nan))
        
        # Get dealer pressure sign - prefer from file, fall back to computed from chain
        dps_val = None
        if dealer_pressure_data:
            dps_val = dealer_pressure_data.get(ticker.upper())
        
        # If not available from file, try to get from computed net_gamma_at_spot
        if dps_val is None:
            dps_val = data.get("net_gamma_at_spot")
        
        dps = dealer_pressure_sign(dps_val)
        
        raw_points.append({
            "ticker": ticker,
            "x": x,
            "y_raw": y_raw,
            "y_display": y_display,
            "spot": data.get("spot"),
            "flip": data.get("flip"),
            "iv": data.get("iv"),
            "hv": data.get("hv20"),
            "atr": data.get("atr14_close_to_close"),
            "put_wall": data.get("walls", {}).get("put_wall"),
            "call_wall": data.get("walls", {}).get("call_wall"),
            "x_nearest_wall_atr": wall_dist.get("x_nearest_wall_atr"),
            "rre_z": rre_z,
            "rre_raw": rre_data.get("rre_raw", float(np.nan)),
            "dps": dps,
            "net_gamma_at_spot": data.get("net_gamma_at_spot"),
            "structural_airspeed": x,
            "load_factor": y_raw,
            "expected_move_5d": data.get("expected_move", {}).get("5d"),
            "expected_move_10d": data.get("expected_move", {}).get("10d"),
            "expected_move_20d": data.get("expected_move", {}).get("20d"),
        })
        points.append({
            "ticker": ticker,
            "x": x,
            "y": y_display,
            "rre_z": rre_z,
            "dps": dps,
        })
        
        # Collect RRE z-scores for top N determination
        if np.isfinite(rre_z):
            rre_z_scores.append((abs(rre_z), ticker))
    
    if skipped_tickers:
        print(f"DEBUG: Skipped {len(skipped_tickers)} tickers: {', '.join(skipped_tickers)}")
    
    if not points:
        print("No valid data points to plot")
        return
    
    print(f"DEBUG: Plotting {len(points)} tickers: {[p['ticker'] for p in points]}")
    
    # Determine top N tickers by |rre_z| for enhanced labeling
    rre_z_scores.sort(reverse=True)
    top_n_tickers = {ticker for _, ticker in rre_z_scores[:rre_top_n]}
    
    # Normalize highlight tickers to uppercase for comparison
    highlight_set = {h.upper() for h in highlight_tickers}
    
    # Check if RRE is effectively available (at least some finite values)
    has_valid_rre = any(np.isfinite(rp.get("rre_z", float(np.nan))) for rp in raw_points)
    
    # For Tier C: identify smallest 30% by |rre_z| with x < 1.0
    rre_abs_values = [(abs(rp.get("rre_z", 0)) if np.isfinite(rp.get("rre_z", float(np.nan))) else 0, i) 
                      for i, rp in enumerate(raw_points)]
    rre_abs_values.sort()
    bottom_30_pct_count = max(1, int(len(rre_abs_values) * 0.3))
    suppress_label_indices = {idx for _, idx in rre_abs_values[:bottom_30_pct_count]}
    
    # Detect if multi-ticker chart
    is_multi_ticker = len(points) > 1
    
    # Set axis ranges before plotting
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 3)
    
    # Add quadrant shading (permission zones) - draw before points so they appear behind
    # Green: x >= 1.5 and y <= 1.2 (room to move, not overloaded)
    ax.axvspan(1.5, 5, ymin=0, ymax=1.2/3.0, alpha=0.12, color='green', zorder=0)
    # Yellow: mid regions (caution zone)
    ax.axvspan(0.8, 1.5, ymin=0, ymax=2.0/3.0, alpha=0.08, color='yellow', zorder=0)
    # Red: x <= 0.8 (stall) or y >= 2.0 (overload)
    ax.axvspan(0, 0.8, ymin=0, ymax=1, alpha=0.12, color='red', zorder=0)
    ax.axhspan(2.0, 3.0, xmin=0, xmax=1, alpha=0.12, color='red', zorder=0)
    
    # Add grid - light grey like reference (above shading)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='lightgrey', zorder=1)
    ax.set_axisbelow(False)  # Grid above shading but below points
    ax.set_facecolor('white')
    
    # Wall lines: only show for single-ticker charts (multi-ticker = clutter)
    if not is_multi_ticker:
        # Single ticker: show wall lines in ATR-distance space
        ref_point = raw_points[0] if raw_points else None
        put_wall_x = None
        call_wall_x = None
        
        if ref_point and ref_point["atr"] and np.isfinite(ref_point["atr"]) and ref_point["atr"] > 0:
            atr = ref_point["atr"]
            spot = ref_point["spot"]
            
            # Convert walls to ATR-distance space: x = |wall - spot| / ATR
            # This makes walls consistent with the x-axis units (ATRs from spot)
            if ref_point["put_wall"] is not None and np.isfinite(spot):
                put_wall_x = abs(spot - ref_point["put_wall"]) / (atr + 1e-9)
            
            if ref_point["call_wall"] is not None and np.isfinite(spot):
                call_wall_x = abs(ref_point["call_wall"] - spot) / (atr + 1e-9)
        
        # Add wall lines (vertical, converted to ATR-distance space) - above grid
        y_max = 3.0
        if put_wall_x is not None and np.isfinite(put_wall_x) and put_wall_x <= 5:
            ax.axvline(put_wall_x, color='lightblue', linestyle='-', linewidth=1.5, alpha=0.8, zorder=2)
            ax.text(put_wall_x, y_max * 0.98, 'put wall', 
                    rotation=90, va='top', ha='right', fontsize=9, color='black', zorder=3)
        
        if call_wall_x is not None and np.isfinite(call_wall_x) and call_wall_x <= 5:
            ax.axvline(call_wall_x, color='lightblue', linestyle='-', linewidth=1.5, alpha=0.8, zorder=2)
            ax.text(call_wall_x, y_max * 0.98, 'call wall', 
                    rotation=90, va='top', ha='right', fontsize=9, color='black', zorder=3)
    
    # Plot points - above everything else
    base_size = 100
    tier_a_annotations = []  # For adjustText if available
    
    for i, p in enumerate(points):
        raw_p = raw_points[i]
        ticker = p["ticker"]
        is_highlighted = ticker.upper() in highlight_set
        
        # Compute marker size from RRE z-score (adjusted scaling)
        rre_z = raw_p.get("rre_z", float(np.nan))
        if np.isfinite(rre_z) and has_valid_rre:
            size_mult = np.clip(1 + 0.25 * rre_z, 0.7, 2.2)
            marker_size = base_size * size_mult
        else:
            marker_size = base_size
        
        # Apply highlight size boost
        if is_highlighted:
            marker_size *= 1.25
        
        # Determine DPS styling (with color tints)
        dps = raw_p.get("dps", "?")
        
        if dps == "+":
            edge_width = 2.0
            edge_color = 'black'
            edge_style = "solid"
            face_tint = (0.85, 1.0, 0.85)  # Slightly greener
        elif dps == "-":
            edge_width = 2.0
            edge_color = 'black'
            edge_style = "solid"
            face_tint = (1.0, 0.85, 0.85)  # Slightly redder
        else:  # "?" - missing/unavailable
            edge_width = 0.7
            edge_color = (0.6, 0.6, 0.6)  # Lighter mid-grey
            edge_style = "solid"
            # Slightly desaturate face tint for missing DPS
            face_tint = (0.85, 0.85, 0.85)  # More desaturated grey
        
        # Plot halo for highlighted tickers (behind main marker)
        if is_highlighted:
            ax.scatter(
                p["x"], p["y"],
                s=marker_size * 1.4,
                facecolor='none',
                edgecolor='white',
                linewidths=1.5,
                alpha=0.7,
                zorder=3
            )
        
        # Determine label tier using new stricter rules
        x = raw_p["x"]
        y_display = raw_p["y_display"]
        wall_dist = raw_p.get("x_nearest_wall_atr")
        rre_z_abs = abs(rre_z) if np.isfinite(rre_z) else 0
        
        # Tier A: FULL LABEL (multiline) if any of:
        is_tier_a = (
            is_highlighted or
            ticker in top_n_tickers or
            x >= 2.5 or
            y_display >= 2.0 or
            dps in ("+", "-")  # DPS is known
        )
        
        # Tier B: COMPACT LABEL ("TICKER" only) if any of:
        is_tier_b = (
            x >= 1.8 or
            rre_z_abs >= 0.75 or
            (wall_dist is not None and np.isfinite(wall_dist) and wall_dist >= 1.0)
        )
        
        # Tier C: NO LABEL if:
        is_tier_c = (
            not is_tier_a and
            not is_tier_b and
            (x < 1.2 and (not np.isfinite(rre_z) or rre_z_abs < 0.5) and not is_highlighted)
        )
        
        # Determine marker alpha based on label tier
        if is_tier_a:
            marker_alpha = 0.95
        elif is_tier_b:
            marker_alpha = 0.7
        else:  # Tier C (no label)
            marker_alpha = 0.4  # Between 0.35-0.5 as requested
        
        # Plot main marker with tier-based alpha
        ax.scatter(
            p["x"], p["y"], 
            s=marker_size, 
            alpha=marker_alpha,
            facecolor=face_tint,
            edgecolors=edge_color, 
            linewidths=edge_width,
            linestyle=edge_style,
            zorder=4
        )
        
        # Build label based on tier - simplified labels
        if is_tier_c:
            label = None  # No label
        elif is_tier_a:
            # Tier A: Ticker + DPS (if available and not "?")
            if dps in ("+", "-"):
                label = f"{ticker}\nDPS:{dps}"
            else:
                label = ticker
        else:  # Tier B
            # Tier B: Just ticker name
            label = ticker
        
        # Add label annotation
        if label:
            bbox_props = None
            fontweight = 'normal'
            if is_highlighted and is_tier_a:
                # Bold label with white bbox for highlighted tickers
                fontweight = 'bold'
                bbox_props = dict(
                    facecolor='white',
                    alpha=0.7,
                    boxstyle='round,pad=0.2',
                    edgecolor='none'
                )
            
            ann = ax.annotate(
                label,
                xy=(p["x"], p["y"]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10 if is_tier_a else 9,
                fontweight=fontweight,
                bbox=bbox_props,
                arrowprops=dict(arrowstyle='->', lw=1, color='black', alpha=0.6),
                zorder=5
            )
            
            if is_tier_a:
                tier_a_annotations.append(ann)
        
        # If load factor was clamped, show red triangle indicator
        if raw_p["y_raw"] > 5.0:
            ax.plot(p["x"], min(p["y"], 2.95), 'r^', markersize=10, alpha=0.8, zorder=5)
    
    # Apply adjustText to Tier A labels if available
    if HAS_ADJUST_TEXT and tier_a_annotations:
        try:
            adjust_text(tier_a_annotations, ax=ax, arrowprops=dict(arrowstyle='->', lw=0.5, alpha=0.3))
        except Exception:
            pass  # Silently fall back to default positioning
    
    # Formatting - fix title overlap
    fig.suptitle("Option Flight Envelope (Snapshot)", fontsize=14, fontweight='bold', y=0.98)
    ax.set_xlabel("Structural Airspeed = |spot - flip| / ATR", fontsize=11)
    ax.set_ylabel("Load Factor = Risk proxy (IV/HV or custom)", fontsize=11)
    ax.set_title("", fontsize=0)  # Clear default title
    
    # Add axis definitions and metrics in top-left (like reference image)
    info_text = []
    # For multi-ticker, show general info; for single-ticker, show ticker-specific metrics
    if is_multi_ticker:
        # Multi-ticker: show general axis definitions only
        info_text.append(f"{len(points)} tickers plotted")
        info_text.append("Structural airspeed x = |spot - flip| / ATR")
        info_text.append("Load proxy y = IV/HV")
        info_text.append("Details saved to CSV")
    elif raw_points:
        # Single-ticker: show ticker-specific metrics
        p = raw_points[0]
        if np.isfinite(p.get("iv")) and np.isfinite(p.get("hv")):
            # Format IV and HV with 2 decimal places
            info_text.append(f"IV={p['iv']:.2f} HV={p['hv']:.2f}")
            # Show raw load factor even if clamped
            if p["y_raw"] > 5.0:
                info_text.append(f"Load Factor: {p['y_raw']:.2f} (clamped at 5.0)")
        info_text.append("Structural airspeed x = |spot - flip| / ATR")
        info_text.append("Load proxy y = IV/HV")
    
    if info_text:
        ax.text(0.02, 0.98, '\n'.join(info_text), 
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='left',
                family='monospace')
    
    # Build compact legend inside plot (bottom-right corner)
    legend_elements = []
    
    # Add RRE size samples if valid RRE data exists
    if has_valid_rre:
        # Low RRE sample
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(0.9, 0.9, 0.9), 
                      markersize=6, markeredgecolor='black', markeredgewidth=1,
                      label='Marker size = realized range expansion (RRE z-score)')
        )
    
    # DPS samples with proper styling
    legend_elements.extend([
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(0.85, 1.0, 0.85),
                  markersize=8, markeredgecolor='black', markeredgewidth=2.0,
                  label='DPS + : dealer long gamma (supportive)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(1.0, 0.85, 0.85),
                  markersize=8, markeredgecolor='black', markeredgewidth=2.0,
                  label='DPS − : dealer short gamma (unstable)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(0.85, 0.85, 0.85),
                  markersize=8, markeredgecolor=(0.6, 0.6, 0.6), markeredgewidth=0.7,
                  label='DPS ? : unknown / not provided'),
    ])
    
    # Place legend inside axes at bottom-right corner with compact styling
    ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.98, 0.05),
              ncol=2, fontsize=7, frameon=True, framealpha=0.8,
              borderpad=0.25, labelspacing=0.25, handlelength=1.0, handletextpad=0.35)
    
    # Add snapshot parameters below suptitle (only for single-ticker charts)
    # For multi-ticker, skip ticker-specific parameters (they're in point labels)
    if not is_multi_ticker and raw_points:
        p = raw_points[0]
        params = []
        if np.isfinite(p.get("spot")):
            params.append(f"spot={p['spot']:.2f}")
        if np.isfinite(p.get("flip")):
            params.append(f"flip={p['flip']:.2f}")
        if p.get("put_wall") is not None and p.get("call_wall") is not None:
            params.append(f"walls={p['put_wall']:.0f}/{p['call_wall']:.0f}")
        
        if params:
            fig.text(0.5, 0.94, ', '.join(params),
                   fontsize=10,
                   horizontalalignment='center', verticalalignment='bottom')
    elif is_multi_ticker:
        # For multi-ticker, show count instead of ticker-specific params
        fig.text(0.5, 0.94, f"{len(points)} tickers",
               fontsize=10,
               horizontalalignment='center', verticalalignment='bottom')
    
    # Use tight_layout with rect to leave space for suptitle and parameter text
    plt.tight_layout(rect=[0, 0, 1, 0.92])  # Full width since legend is inside axes
    plt.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Flight envelope snapshot saved to {output_path}")
    
    # Save detailed CSV with all metrics
    csv_path = output_path.parent / "flight_envelope_details.csv"
    _save_detailed_csv(raw_points, csv_path)
    print(f"Detailed metrics saved to {csv_path}")


def run_batch(
    tickers: List[str],
    output_json: Path,
    make_charts: bool = False,
    dealer_pressure_path: Optional[Path] = None,
    rre_top_n: int = 8,
    highlight_tickers: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    out_dir = output_json.parent

    all_results: Dict[str, Dict] = {}
    for t in tickers:
        try:
            all_results[t.upper()] = compute_envelope_for_ticker(t, out_dir, make_charts=False)
        except Exception as e:
            all_results[t.upper()] = {"ticker": t.upper(), "error": str(e)}
            print(f"Error processing {t}: {e}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Create flight envelope snapshot chart
    if make_charts:
        snapshot_path = out_dir / "flight_envelope_snapshot.png"
        dealer_pressure_data = load_dealer_pressure(dealer_pressure_path) if dealer_pressure_path else None
        plot_flight_envelope_snapshot(
            all_results, 
            snapshot_path,
            dealer_pressure_data=dealer_pressure_data,
            rre_top_n=rre_top_n,
            highlight_tickers=highlight_tickers or [],
        )

    return all_results


def load_tickers() -> list[str]:
    """Load ticker symbols from shared JSON file."""
    # Path to shared tickers.json file (relative to this script's location)
    script_dir = Path(__file__).parent
    tickers_json = script_dir.parent.parent / "tickers.json"
    
    if not tickers_json.exists():
        raise FileNotFoundError(f"Tickers file not found: {tickers_json}")
    
    with open(tickers_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract just the ticker symbols (first element of each [symbol, type] pair)
    return [ticker[0] for ticker in data["tickers"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate option flight envelope charts with RRE and DPS")
    parser.add_argument("--dealer_pressure_path", type=str, help="Path to JSON/CSV file with dealer pressure data")
    parser.add_argument("--rre_top_n", type=int, default=8, help="Number of top tickers by |rre_z| to show enhanced labels (default: 8)")
    parser.add_argument("--rre_zwin", type=int, default=60, help="Rolling window for RRE z-score (default: 60)")
    parser.add_argument("--atr_n", type=int, default=14, help="ATR period (default: 14)")
    parser.add_argument("--highlight", nargs="+", default=[], help="Additional tickers to always show enhanced labels (e.g., --highlight AMD NVDA)")
    parser.add_argument("--output", type=str, default="output/flight_envelopes.json", help="Output JSON path (default: output/flight_envelopes.json)")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    
    args = parser.parse_args()
    
    # Update module-level config (these are used by compute_envelope_for_ticker)
    DEALER_PRESSURE_PATH = Path(args.dealer_pressure_path) if args.dealer_pressure_path else None
    RRE_TOP_N = args.rre_top_n
    RRE_ZWIN = args.rre_zwin
    ATR_N = args.atr_n
    HIGHLIGHT_TICKERS = args.highlight
    
    # Load tickers from shared tickers.json file
    tickers = load_tickers()
    run_batch(
        tickers, 
        Path(args.output), 
        make_charts=not args.no_charts,
        dealer_pressure_path=DEALER_PRESSURE_PATH,
        rre_top_n=RRE_TOP_N,
        highlight_tickers=HIGHLIGHT_TICKERS,
    )
