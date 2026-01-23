"""
State Machine: Maps Markov/Kelly/Teixiptla signals into coarse-grained market states.

This module provides a rule-based state machine that converts current market signals
into a named MarketState, enabling clear action policies and decision gates.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MarketState(Enum):
    """Coarse-grained market states derived from regime, Kelly, and gate signals."""
    PIN = "PIN"
    RANGE = "RANGE"
    TREND = "TREND"
    RUPTURE_PREP = "RUPTURE_PREP"
    RUPTURE_ACTIVE = "RUPTURE_ACTIVE"
    COOLDOWN = "COOLDOWN"


@dataclass
class MarketSignals:
    """Container for current market signals used to compute state."""
    regime: str                # Teixiptla/Markov regime, e.g. "PIN", "EXPRESSIVE", "RUPTURE_CANDIDATE"
    kelly_fraction: float      # canonical Kelly fraction for this trade (adjusted or fractional)
    gate_state: str            # e.g. "BLOCK", "OPEN", "PARTIAL"
    spread: float              # execution spread fraction, if available
    quality: float             # execution quality, if available
    skew_slope: float          # from skew model, if available
    curvature: float           # from skew model, if available


def compute_market_state(signals: MarketSignals, prev_state: Optional[MarketState] = None) -> MarketState:
    """
    Map current Markov/Kelly/Teixiptla signals into a coarse-grained MarketState.

    This is intentionally rule-based and stateless for now; prev_state is accepted
    to allow future transition logic (cooldowns, hysteresis) but is not required.

    Args:
        signals: Current market signals
        prev_state: Previous state (for future transition logic, currently unused)

    Returns:
        Computed MarketState based on signal rules
    """
    regime = signals.regime.upper()
    gate_state = signals.gate_state.upper()
    k = signals.kelly_fraction

    # Hard PIN + BLOCK, zero Kelly → pinned, no expression
    if gate_state == "BLOCK" and regime == "PIN" and k <= 0.0:
        return MarketState.PIN

    # PIN but some Kelly > 0 → range-y, pinned but with some breathing room
    if regime == "PIN" and k > 0.0:
        return MarketState.RANGE

    # Expressive state with positive Kelly → directional trend permission
    if regime == "EXPRESSIVE" and k > 0.0:
        return MarketState.TREND

    # Rupture candidate with gate open or partial → preparing for rupture
    if regime == "RUPTURE_CANDIDATE" and gate_state in {"OPEN", "PARTIAL"} and k > 0.0:
        # crude split between PREP vs ACTIVE based on Kelly size
        if k >= 0.5:
            return MarketState.RUPTURE_ACTIVE
        return MarketState.RUPTURE_PREP

    # Default fallbacks:
    if gate_state == "BLOCK":
        # Some kind of lockout, treat as cooldown unless hard PIN case already caught
        return MarketState.COOLDOWN

    # Non-blocked but not clearly trend/rupture → treat as RANGE
    return MarketState.RANGE


def describe_actions(state: MarketState) -> str:
    """
    Human-readable action policy for each state.
    Keep this conservative; it is advisory, not binding logic.

    Args:
        state: Current market state

    Returns:
        Action policy description string
    """
    if state == MarketState.PIN:
        return "No directional trades; probes only, no reflexive sleeve."
    if state == MarketState.RANGE:
        return "Light probes or mean-reversion structures; reflexive sleeve generally disabled."
    if state == MarketState.TREND:
        return "Directional entries allowed with modest reflexive sleeve sizing."
    if state == MarketState.RUPTURE_PREP:
        return "Small seed positions and staged reflexive sleeve permitted."
    if state == MarketState.RUPTURE_ACTIVE:
        return "Full reflexive sleeve allowed within Kelly and risk caps."
    if state == MarketState.COOLDOWN:
        return "Stand down; no new risk while system cools off."
    return "No explicit guidance."
