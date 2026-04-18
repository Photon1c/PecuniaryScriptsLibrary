# systems/metrics.py
"""
Compute real-time field metrics from the particle ensemble.
"""
import math
from collections import Counter
from config import STRIKE_OI, CLUSTER_RADIUS
from systems.liquidity_field import y_to_price


def nearest_strike(price: float) -> int:
    strikes = list(STRIKE_OI.keys())
    return min(strikes, key=lambda s: abs(s - price))


def cluster_metrics(particles: list) -> dict:
    """
    Group particles by nearest strike and find the dominant cluster.
    Returns:
        dominant_strike: the most populated strike band
        cluster_fraction: fraction of particles in dominant cluster
        strike_counts: {strike: count}
        mean_price: ensemble mean
        std_price: ensemble std
    """
    prices  = [p.price for p in particles if p.alive]
    if not prices:
        return {}

    strike_hits = Counter(nearest_strike(p) for p in prices)
    dominant    = strike_hits.most_common(1)[0]
    mean_price  = sum(prices) / len(prices)
    variance    = sum((p - mean_price) ** 2 for p in prices) / len(prices)

    return {
        "dominant_strike":   dominant[0],
        "cluster_fraction":  dominant[1] / len(prices),
        "strike_counts":     dict(strike_hits),
        "mean_price":        mean_price,
        "std_price":         math.sqrt(variance),
        "n_active":          len(prices),
    }


def gamma_exposure(price: float, oi_map: dict) -> float:
    """
    Rough proxy for dealer gamma exposure at a given price level.
    Sums call OI + put OI weighted by proximity (ATM gamma is highest).
    """
    total = 0.0
    for strike, (c, p) in oi_map.items():
        dist = abs(strike - price)
        if dist < 20:
            weight = 1.0 / (1.0 + dist)
            total += (c + p) * weight
    return total


def field_pressure(price: float, oi_map: dict) -> tuple:
    """
    Return (call_pressure, put_pressure) at a given price.
    Call pressure = total call OI above current price (ceiling).
    Put pressure  = total put OI below current price (floor).
    """
    call_above = sum(c for s, (c, _) in oi_map.items() if s > price)
    put_below  = sum(p for s, (_, p) in oi_map.items() if s < price)
    return call_above, put_below
