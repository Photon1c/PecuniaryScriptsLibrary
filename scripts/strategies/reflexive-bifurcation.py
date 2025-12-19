#Great strategy to test the IV-regime.md file on
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend from start
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
# --- Option chain IV (uses your existing loader + scanner) ---
from data_loader import load_option_chain_data, get_most_recent_option_date, get_latest_price, load_stock_data
from option_scanner import normalize_chain  # or your normalize_chain location


import sys
import os
import tempfile
import shutil
import time
import threading
import gc
from datetime import datetime
from typing import Dict, List, Optional, Tuple

plt.style.use("default")

try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except ImportError:
    try:
        import imageio
        HAS_IMAGEIO = True
    except ImportError:
        HAS_IMAGEIO = False
        print('Warning: imageio not available, will use ffmpeg directly')

# ==========================================
# PHASE 1: CONFIGURATION BLOCK
# ==========================================

# Multi-ticker configuration
TICKER_CONFIGS = {
    "PLTR": {
        "flip_level": 186.00,
        "put_wall": 180.00,
        "call_wall": 195.00,
        "strike_band": 0.025,  # ±2.5%
        "target_exp": "2025-12-26",  # Priority expiration
        "exp_priority": ["2025-12-26", "2026-01-17"],  # Fallback order
    },
    "SPY": {
        "flip_level": 580.00,
        "put_wall": 570.00,
        "call_wall": 590.00,
        "strike_band": 0.025,
        "target_exp": None,  # Auto-select front week
        "exp_priority": [],  # Use front expiration
    },
    "NVDA": {
        "flip_level": 140.00,
        "put_wall": 135.00,
        "call_wall": 145.00,
        "strike_band": 0.025,
        "target_exp": None,
        "exp_priority": [],
    },
}

# Global settings
USE_CHAIN_IV = True
CHAIN_DATE = None  # None = auto (most recent folder)
GENERATE_PNGS = False  # Set to True for visual output (slower)
OUTPUT_DIR = "bifurcation_signals"

# Simulation parameters (for price path generation if needed)
N_STEPS = 120
np.random.seed(42)

# IV slope threshold for regime classification
IV_SLOPE_THRESHOLD = 0.001

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_near_spot_iv_from_chain(
    ticker: str,
    date: str | None,
    spot: float,
    target_exp: str | None = None,
    strike_pct_band: float = 0.025,
    side: str = "BOTH",
    base_dir: str = "F:/inputs/options/log",
) -> tuple[np.ndarray, list]:
    """
    Pulls a near-spot IV time-series proxy from the current option chain snapshot.
    """
    raw = load_option_chain_data(ticker, date=date, base_dir=base_dir)
    tidy = normalize_chain(raw)

    # Optional side filter
    if side in ("C", "P"):
        tidy = tidy[tidy["side"] == side].copy()

    # Optional expiration filter
    if target_exp is not None:
        try:
            target_dt = pd.to_datetime(target_exp)
            if tidy["expiration"].dtype == 'datetime64[ns]':
                tidy = tidy[tidy["expiration"].dt.date == target_dt.date()].copy()
            else:
                tidy = tidy[tidy["expiration"].astype(str).str.contains(target_exp, case=False, na=False)].copy()
        except Exception as e:
            print(f"Warning: Could not filter expiration '{target_exp}': {e}")
            pass

    lo, hi = spot * (1 - strike_pct_band), spot * (1 + strike_pct_band)
    band = tidy[(tidy["strike"] >= lo) & (tidy["strike"] <= hi)].copy()
    band = band.dropna(subset=["iv"])

    # Convert IV from percentage to decimal if needed
    iv_mean_check = band["iv"].mean()
    if pd.notna(iv_mean_check) and iv_mean_check > 1.5:
        band["iv"] = band["iv"] / 100.0

    # OI-weighted IV is usually more stable than raw mean
    w = band["oi"].fillna(0).clip(lower=0)
    band["w"] = w.where(w > 0, 1.0)

    grp = band.groupby("expiration", dropna=False)
    iv_by_exp = grp.apply(lambda g: np.average(g["iv"].values, weights=g["w"].values)).values
    expirations = list(grp.groups.keys())

    return np.asarray(iv_by_exp, float), expirations


def get_iv_surface_by_strike(
    ticker: str,
    date: str | None,
    target_exp: str | None = None,
    base_dir: str = "F:/inputs/options/log",
) -> pd.DataFrame:
    """
    Retrieve IV by strike for a chosen expiration.
    Returns DataFrame with columns: strike, iv_call, iv_put, iv_atm, oi_call, oi_put
    """
    raw = load_option_chain_data(ticker, date=date, base_dir=base_dir)
    tidy = normalize_chain(raw)
    
    # Filter to target expiration
    if target_exp is not None:
        try:
            target_dt = pd.to_datetime(target_exp)
            if tidy["expiration"].dtype == 'datetime64[ns]':
                tidy = tidy[tidy["expiration"].dt.date == target_dt.date()].copy()
            else:
                tidy = tidy[tidy["expiration"].astype(str).str.contains(target_exp, case=False, na=False)].copy()
        except Exception as e:
            print(f"Warning: Could not filter expiration '{target_exp}': {e}")
    
    # If no target_exp, use front expiration (earliest)
    if target_exp is None and len(tidy) > 0:
        if tidy["expiration"].dtype == 'datetime64[ns]':
            front_exp = tidy["expiration"].min()
            tidy = tidy[tidy["expiration"] == front_exp].copy()
        else:
            exp_counts = tidy["expiration"].value_counts()
            if len(exp_counts) > 0:
                front_exp = exp_counts.index[0]
                tidy = tidy[tidy["expiration"] == front_exp].copy()
    
    # Convert IV from percentage to decimal if needed
    iv_mean_check = tidy["iv"].mean()
    if pd.notna(iv_mean_check) and iv_mean_check > 1.5:
        tidy["iv"] = tidy["iv"] / 100.0
    
    # Separate calls and puts
    calls = tidy[tidy["side"] == "C"].copy()
    puts = tidy[tidy["side"] == "P"].copy()
    
    # Create IV surface DataFrame
    all_strikes = sorted(set(calls["strike"].dropna().tolist() + puts["strike"].dropna().tolist()))
    
    surface = pd.DataFrame({"strike": all_strikes})
    
    # Merge call IV (aggregate if multiple rows per strike)
    call_iv = calls[["strike", "iv", "oi"]].copy()
    call_iv = call_iv.groupby("strike").agg({
        "iv": "mean",
        "oi": "sum"
    }).reset_index()
    call_iv = call_iv.rename(columns={"iv": "iv_call", "oi": "oi_call"})
    surface = surface.merge(call_iv, on="strike", how="left")
    
    # Merge put IV (aggregate if multiple rows per strike)
    put_iv = puts[["strike", "iv", "oi"]].copy()
    put_iv = put_iv.groupby("strike").agg({
        "iv": "mean",
        "oi": "sum"
    }).reset_index()
    put_iv = put_iv.rename(columns={"iv": "iv_put", "oi": "oi_put"})
    surface = surface.merge(put_iv, on="strike", how="left")
    
    # Compute ATM IV (average of call and put, or whichever is available)
    surface["iv_atm"] = surface[["iv_call", "iv_put"]].mean(axis=1, skipna=True)
    
    # Fill missing OI with 0
    surface["oi_call"] = surface["oi_call"].fillna(0)
    surface["oi_put"] = surface["oi_put"].fillna(0)
    
    # Sort by strike
    surface = surface.sort_values("strike").reset_index(drop=True)
    
    return surface


def interpolate_iv_from_surface(price: float, iv_surface: pd.DataFrame) -> float:
    """
    Interpolate IV from IV surface based on current price.
    Maps price → nearest strike IV (or interpolates between strikes).
    """
    if len(iv_surface) == 0:
        return 0.50  # fallback
    
    strikes = iv_surface["strike"].values
    iv_atm = iv_surface["iv_atm"].values
    
    # Remove NaN values
    valid_mask = ~np.isnan(iv_atm)
    if not valid_mask.any():
        return 0.50  # fallback
    
    strikes_valid = strikes[valid_mask]
    iv_atm_valid = iv_atm[valid_mask]
    
    # If price is outside range, use boundary value
    if price <= strikes_valid[0]:
        return iv_atm_valid[0]
    if price >= strikes_valid[-1]:
        return iv_atm_valid[-1]
    
    # Interpolate using numpy
    iv_interp = np.interp(price, strikes_valid, iv_atm_valid)
    
    return float(iv_interp)


def compute_iv_skew(iv_surface: pd.DataFrame, spot: float, strike_band: float = 0.03) -> dict:
    """
    Compute IV skew metrics:
    - iv_atm: near-ATM IV (mean or OI-weighted mean in ±2–3% band)
    - iv_put_skew: slope of IV vs strike for puts (below ATM)
    - iv_call_skew: slope of IV vs strike for calls (above ATM)
    """
    lo, hi = spot * (1 - strike_band), spot * (1 + strike_band)
    band = iv_surface[(iv_surface["strike"] >= lo) & (iv_surface["strike"] <= hi)].copy()
    
    if len(band) == 0:
        return {"iv_atm": 0.50, "iv_put_skew": 0.0, "iv_call_skew": 0.0}
    
    # OI-weighted ATM IV
    w_call = band["oi_call"].fillna(0).clip(lower=0)
    w_put = band["oi_put"].fillna(0).clip(lower=0)
    w_total = w_call + w_put
    w_total = w_total.where(w_total > 0, 1.0)
    
    iv_atm = np.average(band["iv_atm"].values, weights=w_total.values)
    
    # Compute skew (slope of IV vs strike)
    puts_below = iv_surface[(iv_surface["strike"] < spot) & (iv_surface["iv_put"].notna())].copy()
    if len(puts_below) >= 2:
        x = puts_below["strike"].values
        y = puts_below["iv_put"].values
        if len(x) > 0 and np.std(x) > 0:
            iv_put_skew = np.polyfit(x, y, 1)[0]
        else:
            iv_put_skew = 0.0
    else:
        iv_put_skew = 0.0
    
    calls_above = iv_surface[(iv_surface["strike"] > spot) & (iv_surface["iv_call"].notna())].copy()
    if len(calls_above) >= 2:
        x = calls_above["strike"].values
        y = calls_above["iv_call"].values
        if len(x) > 0 and np.std(x) > 0:
            iv_call_skew = np.polyfit(x, y, 1)[0]
        else:
            iv_call_skew = 0.0
    else:
        iv_call_skew = 0.0
    
    return {
        "iv_atm": float(iv_atm),
        "iv_put_skew": float(iv_put_skew),
        "iv_call_skew": float(iv_call_skew)
    }


def detect_vol_pocket(iv_surface: pd.DataFrame, spot: float, threshold: float = 0.02) -> dict:
    """
    Detect "vol pocket" vs directional skew.
    
    Vol pocket: IV decreases symmetrically for both calls and puts above spot.
    This means the market is comfortable with price oscillating in that zone.
    
    Returns:
    - is_vol_pocket: bool
    - upside_iv_compression: float (IV drop above spot, negative = compression)
    - downside_iv_expansion: float (IV increase below spot, positive = expansion)
    """
    # Get IV at spot
    iv_at_spot = interpolate_iv_from_surface(spot, iv_surface)
    
    # Sample strikes above spot (e.g., +2.5%, +5%, +7.5%)
    strikes_above = [spot * 1.025, spot * 1.05, spot * 1.075]
    iv_above = [interpolate_iv_from_surface(s, iv_surface) for s in strikes_above]
    
    # Sample strikes below spot (e.g., -2.5%, -5%, -7.5%)
    strikes_below = [spot * 0.975, spot * 0.95, spot * 0.925]
    iv_below = [interpolate_iv_from_surface(s, iv_surface) for s in strikes_below]
    
    # Compute average IV change above spot
    if len(iv_above) > 0 and all(not np.isnan(v) for v in iv_above):
        avg_iv_above = np.mean([v for v in iv_above if not np.isnan(v)])
        upside_compression = avg_iv_above - iv_at_spot  # Negative = compression
    else:
        upside_compression = 0.0
    
    # Compute average IV change below spot
    if len(iv_below) > 0 and all(not np.isnan(v) for v in iv_below):
        avg_iv_below = np.mean([v for v in iv_below if not np.isnan(v)])
        downside_expansion = avg_iv_below - iv_at_spot  # Positive = expansion
    else:
        downside_expansion = 0.0
    
    # Vol pocket detection: both calls and puts show IV compression above spot
    # Check if IV decreases above spot for both call and put sides
    calls_above = iv_surface[(iv_surface["strike"] > spot) & (iv_surface["iv_call"].notna())].copy()
    puts_above = iv_surface[(iv_surface["strike"] > spot) & (iv_surface["iv_put"].notna())].copy()
    
    call_compression = False
    put_compression = False
    
    if len(calls_above) >= 2:
        # Check if call IV decreases as strike increases
        call_slope = np.polyfit(calls_above["strike"].values, calls_above["iv_call"].values, 1)[0]
        call_compression = call_slope < -threshold
    
    if len(puts_above) >= 2:
        # Check if put IV decreases as strike increases (above spot)
        put_slope = np.polyfit(puts_above["strike"].values, puts_above["iv_put"].values, 1)[0]
        put_compression = put_slope < -threshold
    
    is_vol_pocket = call_compression and put_compression
    
    return {
        "is_vol_pocket": is_vol_pocket,
        "upside_iv_compression": float(upside_compression),
        "downside_iv_expansion": float(downside_expansion),
        "call_compression_above_spot": call_compression,
        "put_compression_above_spot": put_compression,
    }


# ==========================================
# PHASE 2: SIGNAL EXTRACTION
# ==========================================

def extract_signals(
    ticker: str,
    current_price: float,
    config: Dict,
    chain_date: Optional[str] = None,
) -> Dict:
    """
    Extract signals for a single ticker.
    Returns:
    - price_state: "above_flip" | "below_flip"
    - iv_state: "rising" | "flat" | "compressing"
    - iv_level: float (current IV as decimal)
    - iv_slope: float
    - bifurcation_risk: int (0-100)
    - suggested_structure: "defensive" | "neutral" | "opportunistic"
    """
    flip_level = config["flip_level"]
    put_wall = config["put_wall"]
    call_wall = config["call_wall"]
    strike_band = config["strike_band"]
    target_exp = config.get("target_exp")
    
    # Price state
    if current_price >= flip_level:
        price_state = "above_flip"
    else:
        price_state = "below_flip"
    
    # Get IV data
    try:
        if chain_date is None:
            chain_date = get_most_recent_option_date(ticker)
        
        iv_surface = get_iv_surface_by_strike(
            ticker=ticker,
            date=chain_date,
            target_exp=target_exp,
        )
        
        if len(iv_surface) == 0:
            raise ValueError("IV surface is empty")
        
        # Interpolate IV at current price
        iv_level = interpolate_iv_from_surface(current_price, iv_surface)
        
        # Compute IV slope (use recent price history if available, otherwise use surface gradient)
        # For now, we'll compute a simple gradient from the IV surface around current price
        price_range = np.linspace(current_price * 0.98, current_price * 1.02, 10)
        iv_range = [interpolate_iv_from_surface(p, iv_surface) for p in price_range]
        iv_slope = np.gradient(iv_range)[len(iv_range)//2]  # Slope at center
        
        # IV state classification
        if iv_slope > IV_SLOPE_THRESHOLD:
            iv_state = "rising"
        elif iv_slope < -IV_SLOPE_THRESHOLD:
            iv_state = "compressing"
        else:
            iv_state = "flat"
        
        # Compute skew metrics
        skew_metrics = compute_iv_skew(iv_surface, current_price, strike_band)
        
        # Detect vol pocket vs directional skew
        vol_pocket_metrics = detect_vol_pocket(iv_surface, current_price, threshold=0.02)
        
        # Compute upside convexity score
        # If IV decreases above spot symmetrically → LOW convexity (range-bound expected)
        if vol_pocket_metrics["is_vol_pocket"]:
            upside_convexity_score = "LOW"  # Vol pocket = range-bound, not explosive
            range_probability = "HIGH"
        elif vol_pocket_metrics["upside_iv_compression"] < -0.01:  # Significant compression
            upside_convexity_score = "LOW"
            range_probability = "MEDIUM"
        elif vol_pocket_metrics["upside_iv_compression"] > 0.01:  # IV expansion above spot
            upside_convexity_score = "HIGH"  # Explosive upside possible
            range_probability = "LOW"
        else:
            upside_convexity_score = "MEDIUM"
            range_probability = "MEDIUM"
        
        # Preferred structure recommendation
        if upside_convexity_score == "LOW" and range_probability == "HIGH":
            # Upside = carry, not convexity → favor spreads, short vol
            preferred_structure = "spreads_short_vol"
            structure_note = "Upside expected range-bound; favor call spreads over naked calls"
        elif upside_convexity_score == "HIGH":
            # High convexity → can pay for gamma
            preferred_structure = "long_convexity"
            structure_note = "High upside convexity; naked calls or call spreads viable"
        else:
            # Medium → neutral structures
            preferred_structure = "neutral"
            structure_note = "Moderate convexity; structure based on directional view"
        
    except Exception as e:
        print(f"  ⚠ Error loading IV for {ticker}: {e}")
        iv_level = 0.50
        iv_slope = 0.0
        iv_state = "flat"
        skew_metrics = {"iv_atm": 0.50, "iv_put_skew": 0.0, "iv_call_skew": 0.0}
        vol_pocket_metrics = {
            "is_vol_pocket": False,
            "upside_iv_compression": 0.0,
            "downside_iv_expansion": 0.0,
        }
        upside_convexity_score = "MEDIUM"
        range_probability = "MEDIUM"
        preferred_structure = "neutral"
        structure_note = "IV data unavailable"
    
    # Bifurcation risk score (0-100)
    # Factors:
    # - Price proximity to walls (higher risk near walls)
    # - IV state (rising IV = higher risk)
    # - Price below flip (higher risk)
    
    risk_score = 0
    
    # Price proximity risk (0-40 points)
    if current_price <= put_wall:
        risk_score += 40  # At or below put wall
    elif current_price <= flip_level:
        risk_score += 20 + 20 * (1 - (current_price - put_wall) / (flip_level - put_wall))
    elif current_price >= call_wall:
        risk_score += 30  # At or above call wall
    elif current_price >= flip_level:
        risk_score += 10 * (current_price - flip_level) / (call_wall - flip_level)
    
    # IV state risk (0-30 points)
    if iv_state == "rising":
        risk_score += 30
    elif iv_state == "flat":
        risk_score += 15
    else:  # compressing
        risk_score += 5
    
    # Price state risk (0-30 points)
    if price_state == "below_flip":
        risk_score += 30
    else:
        risk_score += 10
    
    risk_score = min(100, max(0, int(risk_score)))
    
    # Suggested structure
    if risk_score >= 70:
        suggested_structure = "defensive"
    elif risk_score >= 40:
        suggested_structure = "neutral"
    else:
        suggested_structure = "opportunistic"
    
    return {
        "ticker": ticker,
        "price": current_price,
        "price_state": price_state,
        "iv_level": iv_level,
        "iv_state": iv_state,
        "iv_slope": iv_slope,
        "bifurcation_risk": risk_score,
        "suggested_structure": suggested_structure,
        "flip_level": flip_level,
        "put_wall": put_wall,
        "call_wall": call_wall,
        "iv_put_skew": skew_metrics["iv_put_skew"],
        "iv_call_skew": skew_metrics["iv_call_skew"],
        # Vol pocket and convexity metrics
        "is_vol_pocket": vol_pocket_metrics["is_vol_pocket"],
        "upside_iv_compression": vol_pocket_metrics["upside_iv_compression"],
        "downside_iv_expansion": vol_pocket_metrics["downside_iv_expansion"],
        "upside_convexity_score": upside_convexity_score,
        "range_probability": range_probability,
        "preferred_structure": preferred_structure,
        "structure_note": structure_note,
    }


# ==========================================
# PHASE 3: OUTPUT FUNCTIONS
# ==========================================

def print_terminal_table(signals: List[Dict]):
    """Print fast terminal table output."""
    print("\n" + "="*120)
    print("BIFURCATION SIGNAL EXTRACTION")
    print("="*120)
    print(f"{'Ticker':<8} {'Price':<8} {'IV':<8} {'IV State':<12} {'Risk':<6} {'Convexity':<10} {'Range':<8} {'Structure':<20}")
    print("-"*120)
    
    for sig in signals:
        iv_pct = f"{sig['iv_level']*100:.1f}%"
        vol_pocket_marker = "🔵" if sig.get('is_vol_pocket', False) else "  "
        print(f"{sig['ticker']:<8} {sig['price']:<8.2f} {iv_pct:<8} {sig['iv_state']:<12} "
              f"{sig['bifurcation_risk']:<6} {sig.get('upside_convexity_score', 'N/A'):<10} "
              f"{sig.get('range_probability', 'N/A'):<8} {vol_pocket_marker} {sig.get('preferred_structure', 'N/A'):<20}")
    
    print("-"*120)
    print("🔵 = Vol Pocket detected (symmetric IV compression above spot)")
    print("="*120 + "\n")
    
    # Detailed notes for each ticker
    for sig in signals:
        if sig.get('structure_note'):
            print(f"{sig['ticker']}: {sig['structure_note']}")
            if sig.get('is_vol_pocket'):
                print(f"  → Vol pocket: Market comfortable with price oscillating in {sig['price']:.2f} zone")
                print(f"  → Upside IV compression: {sig.get('upside_iv_compression', 0)*100:.2f}%")
            print()


def save_csv_log(signals: List[Dict], filename: str = None):
    """Save signals to CSV for backtesting."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{OUTPUT_DIR}/bifurcation_signals_{timestamp}.csv"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df = pd.DataFrame(signals)
    df["timestamp"] = datetime.now()
    df.to_csv(filename, index=False)
    print(f"✓ Saved signals to {filename}")


# ==========================================
# ANIMATION GENERATION (Optional)
# ==========================================

def generate_price_path_simulation(initial_price: float, n_steps: int = 120) -> np.ndarray:
    """
    Generate a simulated price path for visualization.
    Gentle drift up, then breakdown (similar to original).
    """
    price = np.zeros(n_steps)
    price[0] = initial_price
    
    for i in range(1, n_steps):
        if i < 50:
            # Drift regime (favors call)
            price[i] = price[i-1] + 0.12 + np.random.normal(0, 0.25)
        else:
            # Breakdown regime (favors put)
            price[i] = price[i-1] - 0.25 + np.random.normal(0, 0.35)
    
    return price


def generate_iv_path_from_surface(price_path: np.ndarray, iv_surface: pd.DataFrame) -> np.ndarray:
    """Generate IV path by interpolating from surface based on price path."""
    iv_path = np.zeros(len(price_path), dtype=float)
    for i in range(len(price_path)):
        iv_path[i] = interpolate_iv_from_surface(price_path[i], iv_surface)
    return iv_path


def create_animated_chart(ticker: str, config: Dict, chain_date: Optional[str] = None):
    """
    Create animated chart for a specific ticker.
    Generates MP4 video and final PNG frame.
    """
    print(f"\n{'='*100}")
    print(f"Generating animated chart for {ticker}...")
    print(f"{'='*100}")
    
    flip_level = config["flip_level"]
    put_wall = config["put_wall"]
    call_wall = config["call_wall"]
    target_exp = config.get("target_exp")
    
    # Get current price
    try:
        initial_price = get_latest_price(ticker)
    except Exception as e:
        print(f"✗ Error getting price for {ticker}: {e}")
        return
    
    # Generate price path simulation
    print(f"  Generating price path simulation (starting at {initial_price:.2f})...")
    price = generate_price_path_simulation(initial_price, N_STEPS)
    t = np.arange(N_STEPS)
    
    # Get IV surface and generate IV path
    try:
        if chain_date is None:
            chain_date = get_most_recent_option_date(ticker)
        
        iv_surface = get_iv_surface_by_strike(ticker, chain_date, target_exp)
        
        if len(iv_surface) > 0:
            print(f"  Generating IV path from chain surface...")
            iv = generate_iv_path_from_surface(price, iv_surface)
            iv_slope = np.gradient(iv)
        else:
            raise ValueError("IV surface is empty")
    except Exception as e:
        print(f"  ⚠ Error loading IV: {e}, using simulated IV")
        iv_anchor = 0.50
        from scipy import stats
        # Simple simulated IV that rises during breakdown
        iv = np.full(N_STEPS, iv_anchor)
        for i in range(50, N_STEPS):
            iv[i] = iv_anchor + 0.05 * (i - 50) / 70.0
        iv_slope = np.gradient(iv)
    
    # Entry logic (simplified for visualization)
    CALL_ENTRY_IDX = None
    CALL_ENTRY_PRICE_ACTUAL = None
    PUT_ENTRY_IDX = None
    PUT_ENTRY_PRICE_ACTUAL = None
    
    # Simple entry: call at peak, put at breakdown
    peak_idx = np.argmax(price[:50])
    if peak_idx > 0:
        PUT_ENTRY_IDX = peak_idx
        PUT_ENTRY_PRICE_ACTUAL = price[peak_idx]
    
    # Animation update function
    def update(frame: int, line_price, line_flip, line_pw, line_cw,
               call_marker, put_marker, line_iv, text_state,
               call_entry_line=None, put_entry_line=None):
        line_price.set_data(t[:frame+1], price[:frame+1])
        
        # Color by IV regime
        if frame > 0:
            iv_slope_now = iv_slope[frame] if frame < len(iv_slope) else 0.0
            if iv_slope_now > IV_SLOPE_THRESHOLD:
                line_price.set_color('#d62728')
            elif iv_slope_now < -IV_SLOPE_THRESHOLD:
                line_price.set_color('#2ca02c')
            else:
                line_price.set_color('#1f77b4')
        
        line_flip.set_data(t[:frame+1], np.full(frame+1, flip_level))
        line_pw.set_data(t[:frame+1], np.full(frame+1, put_wall))
        line_cw.set_data(t[:frame+1], np.full(frame+1, call_wall))
        line_iv.set_data(t[:frame+1], iv[:frame+1])
        
        if PUT_ENTRY_IDX is not None and frame >= PUT_ENTRY_IDX:
            put_marker.set_data([PUT_ENTRY_IDX], [PUT_ENTRY_PRICE_ACTUAL])
        else:
            put_marker.set_data([], [])
        
        spot = price[frame]
        iv_now = iv[frame]
        iv_slope_now = iv_slope[frame] if frame < len(iv_slope) else 0.0
        
        if iv_slope_now > IV_SLOPE_THRESHOLD:
            iv_regime = "IV Rising"
        elif iv_slope_now < -IV_SLOPE_THRESHOLD:
            iv_regime = "IV Compressing"
        else:
            iv_regime = "IV Flat"
        
        if frame < 50:
            price_regime = "Drift Regime (Calm, Call-Favored)"
        else:
            price_regime = "Breakdown Regime (Vol Rising, Put-Favored)"
        
        strat_state = "No legs active yet."
        if PUT_ENTRY_IDX is not None and frame >= PUT_ENTRY_IDX:
            strat_state = f"Put leg active @ {PUT_ENTRY_PRICE_ACTUAL:.2f}"
        
        text_state.set_text(
            f"t={frame} | Spot={spot:.2f} | Flip={flip_level:.2f} | "
            f"IV={iv_now*100:.1f}% ({iv_regime}) | {price_regime} | {strat_state}"
        )
        
        return (line_price, line_flip, line_pw, line_cw, call_marker, put_marker, line_iv, text_state)
    
    # Generate animation
    print('  Starting animation save (frame-by-frame method)...')
    print('  Note: This may take a minute. Progress will be shown below.')
    
    temp_dir = tempfile.mkdtemp(prefix='rbs_frames_')
    frame_files = []
    
    try:
        fig_save, (ax_price_save, ax_iv_save) = plt.subplots(
            2, 1, figsize=(8, 6), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]}
        )
        
        ax_price_save.set_title(f"Reflexive Bifurcation Strategy – Price-Level Entry System ({ticker})")
        ax_price_save.set_ylabel(f"{ticker} Price")
        ax_price_save.grid(True, alpha=0.3)
        ax_iv_save.set_xlabel("Time (arbitrary units)")
        ax_iv_save.set_ylabel("IV")
        ax_iv_save.grid(True, alpha=0.3)
        
        line_price_save, = ax_price_save.plot([], [], lw=2, label="Spot", color='#1f77b4')
        line_flip_save,  = ax_price_save.plot([], [], "k--", lw=1, label="Flip")
        line_pw_save,    = ax_price_save.plot([], [], "g--", lw=1, label="Put Wall")
        line_cw_save,    = ax_price_save.plot([], [], "r--", lw=1, label="Call Wall")
        put_marker_save, = ax_price_save.plot([], [], marker="v", ms=10, color="tab:red",
                                              linestyle="None", label="Put Entry")
        line_iv_save, = ax_iv_save.plot([], [], color="tab:purple", lw=2, label="IV")
        
        handles_price_save, labels_price_save = ax_price_save.get_legend_handles_labels()
        handles_iv_save, labels_iv_save = ax_iv_save.get_legend_handles_labels()
        fig_save.legend(
            handles_price_save + handles_iv_save,
            labels_price_save + labels_iv_save,
            loc="lower center", bbox_to_anchor=(0.5, 0.04),
            ncol=4, frameon=True, fontsize=8
        )
        
        text_state_save = fig_save.text(
            0.5, 0.01, "", ha="center", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )
        fig_save.subplots_adjust(hspace=0.15, bottom=0.20, top=0.92)
        
        ax_price_save.set_xlim(0, N_STEPS)
        ax_price_save.set_ylim(min(price) - 3, max(price) + 3)
        ax_iv_save.set_xlim(0, N_STEPS)
        ax_iv_save.set_ylim(iv.min() - 0.005, iv.max() + 0.005)
        
        print("  Rendering frames...")
        for frame in range(N_STEPS):
            update(frame, line_price_save, line_flip_save, line_pw_save,
                   line_cw_save, None, put_marker_save,
                   line_iv_save, text_state_save)
            
            frame_path = os.path.join(temp_dir, f"frame_{frame:04d}.png")
            fig_save.savefig(frame_path, dpi=120, bbox_inches='tight', facecolor='white')
            frame_files.append(frame_path)
            
            if (frame + 1) % 5 == 0:
                gc.collect()
            
            if (frame + 1) % 10 == 0 or frame == 0:
                print(f"    Rendered frame {frame + 1} of {N_STEPS}", flush=True)
        
        print(f"\n  ✓ All {N_STEPS} frames rendered successfully.")
        
        # Combine frames with ffmpeg
        output_video = f"{ticker}_bifurcation_demo.mp4"
        output_png = f"{ticker}_bifurcation_final.png"
        
        print("  Combining frames into video using ffmpeg...")
        try:
            import subprocess
            cmd = [
                'ffmpeg', '-y',
                '-framerate', '15',
                '-i', os.path.join(temp_dir, 'frame_%04d.png'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '23',
                '-preset', 'medium',
                output_video
            ]
            
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1
            )
            stdout, stderr = process.communicate(timeout=120)
            
            if process.returncode == 0:
                print(f"  ✓ Animation saved as {output_video}")
            else:
                print(f"  ✗ ffmpeg error (return code {process.returncode})")
        except FileNotFoundError:
            print("  ✗ ffmpeg not found. Install ffmpeg to create video output.")
        except Exception as e:
            print(f'  ✗ Error: {e}')
        
        # Save final frame
        try:
            shutil.copy(frame_files[-1], output_png)
            print(f"  ✓ Final frame saved as {output_png}")
        except Exception as e:
            print(f"  ✗ Error saving final frame: {e}")
        
        plt.close(fig_save)
        
    finally:
        try:
            shutil.rmtree(temp_dir)
            print(f"  ✓ Cleaned up temporary files.")
        except Exception as e:
            print(f"  Warning: Could not clean up {temp_dir}: {e}")


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # Get current prices from data files
    all_signals = []
    
    print("Extracting bifurcation signals...")
    print("="*100)
    
    for ticker in TICKER_CONFIGS.keys():
        config = TICKER_CONFIGS[ticker]
        
        try:
            # Get current price
            print(f"\nProcessing {ticker}...")
            current_price = get_latest_price(ticker)
            
            # Extract signals
            signals = extract_signals(ticker, current_price, config, CHAIN_DATE)
            all_signals.append(signals)
            
        except Exception as e:
            print(f"  ✗ Error processing {ticker}: {e}")
            continue
    
    # Phase 3: Output
    if len(all_signals) > 0:
        print_terminal_table(all_signals)
        save_csv_log(all_signals)
        
        # Ask user if they want to generate animated chart
        print("\n" + "="*100)
        print("ANIMATED CHART GENERATION")
        print("="*100)
        print("Available tickers:", ", ".join(TICKER_CONFIGS.keys()))
        print("\nEnter ticker to generate animated chart (or press Enter to skip): ", end="")
        
        try:
            user_input = input().strip().upper()
            
            if user_input and user_input in TICKER_CONFIGS:
                create_animated_chart(user_input, TICKER_CONFIGS[user_input], CHAIN_DATE)
            elif user_input:
                print(f"  ✗ '{user_input}' not found in configured tickers.")
            else:
                print("  Skipping chart generation.")
        except (EOFError, KeyboardInterrupt):
            print("\n  Skipping chart generation.")
    else:
        print("\n⚠ No signals extracted. Check ticker configurations and data availability.")
