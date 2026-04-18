# systems/liquidity_field.py
"""
Computes the net gravitational acceleration on a price particle
due to all open-interest wells.

Coordinate convention:
  - screen Y increases DOWNWARD
  - price increases UPWARD  → higher price = smaller Y on screen
"""
import math
from config import (
    STRIKE_OI, SPY_RANGE_LOW, SPY_RANGE_HIGH,
    GRAVITY_FALLOFF, CALL_GRAVITY, PUT_GRAVITY,
    COMBINED_SCALE, MAX_ACCEL, CHART_MARGIN,
)
from utils.vectors import clamp


def price_to_y(price: float, height: int, margin: int = CHART_MARGIN) -> float:
    """Map a price to a screen Y coordinate."""
    usable = height - 2 * margin
    frac = (price - SPY_RANGE_LOW) / (SPY_RANGE_HIGH - SPY_RANGE_LOW)
    return height - margin - frac * usable   # invert: higher price → smaller Y


def y_to_price(y: float, height: int, margin: int = CHART_MARGIN) -> float:
    """Map a screen Y coordinate back to a price."""
    usable = height - 2 * margin
    frac = (height - margin - y) / usable
    return SPY_RANGE_LOW + frac * (SPY_RANGE_HIGH - SPY_RANGE_LOW)


def net_acceleration(price: float) -> float:
    """
    Return net vertical acceleration (in price units per tick²) at `price`.

    Call OI above current price acts as a ceiling (repels price upward → negative accel when price above strike).
    Put OI below current price acts as a floor (repels price downward → positive accel when price below strike).
    Both also attract price toward the strike when price is away from it.

    Simplified model:
      - Each strike exerts a signed force ∝ OI / distance^falloff
      - Calls: attract price UP toward strike when below, repel DOWN when above
      - Puts:  attract price DOWN toward strike when above, repel UP when below
    """
    accel = 0.0
    for strike, (call_oi, put_oi) in STRIKE_OI.items():
        delta = strike - price   # positive = strike is above current price
        dist  = max(abs(delta), 0.5)
        weight = 1.0 / (dist ** GRAVITY_FALLOFF)

        # Calls: attract price toward strike from below, resist from above
        # Net: force is in the direction of delta (toward the strike) scaled by call OI
        call_force = CALL_GRAVITY * call_oi * weight * (1.0 if delta > 0 else -1.0)

        # Puts: attract price toward strike from above, resist from below
        # Net: force is in the direction of delta (toward the strike) scaled by put OI
        put_force  = PUT_GRAVITY  * put_oi  * weight * (1.0 if delta > 0 else -1.0)

        # Total combined — calls pull harder from below, puts pull harder from above
        total = (call_force + put_force) * COMBINED_SCALE
        accel += total

    return clamp(accel, -MAX_ACCEL, MAX_ACCEL)


def build_field_cache(height: int, margin: int = CHART_MARGIN) -> list:
    """
    Precompute net acceleration for every screen Y pixel.
    Returns list of (y, price, accel) tuples.
    """
    cache = []
    for y in range(margin, height - margin):
        p = y_to_price(y, height, margin)
        a = net_acceleration(p)
        cache.append((y, p, a))
    return cache


def compute_max_pain(oi_map: dict) -> float:
    """
    Classic max pain: strike where total dollar value of expiring options
    is minimized for buyers (maximized loss for retail).
    Approximate: minimize sum over all strikes of OI × |strike - candidate|.
    """
    strikes = list(oi_map.keys())
    best_strike = strikes[0]
    best_pain   = float('inf')
    for candidate in strikes:
        pain = sum(
            (call_oi + put_oi) * abs(s - candidate)
            for s, (call_oi, put_oi) in oi_map.items()
        )
        if pain < best_pain:
            best_pain   = pain
            best_strike = candidate
    return float(best_strike)
