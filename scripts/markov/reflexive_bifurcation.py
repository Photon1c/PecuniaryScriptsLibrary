"""
Reflexive Bifurcation Sleeve: Nested leg planning for option strategies.

This module generates a nested reflexive bifurcation sleeve plan that allocates
capital across multiple option legs with progressive sizing and stop-loss rules.
"""

from dataclasses import dataclass
from typing import Literal

# Type alias for option direction
Direction = Literal["call", "put"]


@dataclass
class LegPlan:
    """Plan for a single leg in a reflexive bifurcation sleeve."""
    leg: int
    direction: Direction
    dte: float  # days to expiration
    sleeve_entry: float  # capital allocated to this leg
    stop_loss: float  # dollar stop for this leg


def generate_reflexive_plan(
    K: float,
    kelly_fraction: float,
    exp_cap_frac: float = 0.20,
    stop_frac: float = 0.097,
    dte_initial: float = 2.0,
    initial_direction: str = "call",
    max_legs: int = 2,
) -> list[LegPlan]:
    """
    Build a nested reflexive bifurcation sleeve plan.

    Args:
        K: Total portfolio capital.
        kelly_fraction: Kelly fraction for this trade (0–1).
        exp_cap_frac: Max fraction of K available to this sleeve (cap).
        stop_frac: Fractional stop per leg (e.g. 0.097 = 9.7%).
        dte_initial: Initial days to expiration.
        initial_direction: "call" or "put".
        max_legs: Number of nested legs to pre-plan.

    Returns:
        List of LegPlan objects, one per leg.

    Logic:
        - Compute sleeve cap E0 = exp_cap_frac * K.
        - Actual sleeve E0_eff = min(E0, kelly_fraction * K).
          If kelly_fraction == 0, return an empty plan.
        - For leg n:
            - leg 1: sleeve = E0_eff, dte = dte_initial, direction = initial_direction
            - stop_loss = stop_frac * sleeve
            - append LegPlan(leg, direction, dte, rounded sleeve, rounded stop_loss)
            - next sleeve = sleeve * (1 - stop_frac)
            - next dte = dte * 2.0
            - next direction = flipped ("call" <-> "put")
        - Validate arguments (K > 0, 0 < exp_cap_frac <= 1, 0 < stop_frac < 1, max_legs >= 1).
    """
    # Validate arguments
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")
    if not (0 < exp_cap_frac <= 1):
        raise ValueError(f"exp_cap_frac must be in (0, 1], got {exp_cap_frac}")
    if not (0 < stop_frac < 1):
        raise ValueError(f"stop_frac must be in (0, 1), got {stop_frac}")
    if max_legs < 1:
        raise ValueError(f"max_legs must be >= 1, got {max_legs}")
    if initial_direction not in ("call", "put"):
        raise ValueError(f"initial_direction must be 'call' or 'put', got {initial_direction}")

    # If kelly_fraction is zero or negative, return empty plan
    if kelly_fraction <= 0.0:
        return []

    # Compute sleeve cap E0 = exp_cap_frac * K
    E0 = exp_cap_frac * K

    # Actual sleeve E0_eff = min(E0, kelly_fraction * K)
    E0_eff = min(E0, kelly_fraction * K)

    # If effective sleeve is zero or negative, return empty plan
    if E0_eff <= 0:
        return []

    plan = []
    current_sleeve = E0_eff
    current_dte = dte_initial
    current_direction: Direction = initial_direction  # type: ignore

    for leg_num in range(1, max_legs + 1):
        # Calculate stop loss for this leg
        stop_loss = stop_frac * current_sleeve

        # Round to 2 decimal places for practical use
        sleeve_rounded = round(current_sleeve, 2)
        stop_rounded = round(stop_loss, 2)

        # Create leg plan
        leg_plan = LegPlan(
            leg=leg_num,
            direction=current_direction,
            dte=current_dte,
            sleeve_entry=sleeve_rounded,
            stop_loss=stop_rounded
        )
        plan.append(leg_plan)

        # Prepare for next leg:
        # - Next sleeve = current_sleeve * (1 - stop_frac)
        current_sleeve = current_sleeve * (1 - stop_frac)
        # - Next dte = current_dte * 2.0
        current_dte = current_dte * 2.0
        # - Next direction = flipped ("call" <-> "put")
        current_direction = "put" if current_direction == "call" else "call"

    return plan
