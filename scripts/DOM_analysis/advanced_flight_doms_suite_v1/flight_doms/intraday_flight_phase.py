"""
intraday_flight_phase.py

Scale-aware intraday "flight phase" classifier for SPY (or any ticker).
Designed to be called from a DOM/price polling loop.

Core idea
---------
We maintain a rolling buffer of (timestamp, price, bid_size, ask_size)
and classify the last N seconds into:

    - TAKEOFF
    - CLIMB
    - CRUISE
    - DESCENT
    - HOLDING
    - TURBULENCE
    - WARMUP (not enough data yet)

You can tune thresholds per symbol.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Optional, Literal, Dict, Any
import time
import math


FlightPhase = Literal[
    "WARMUP",
    "TAKEOFF",
    "CLIMB",
    "CRUISE",
    "DESCENT",
    "HOLDING",
    "TURBULENCE",
]


@dataclass
class Sample:
    """Single snapshot of price + DOM sizes."""
    t: float           # Unix timestamp in seconds
    price: float
    bid_size: float
    ask_size: float


@dataclass
class PhaseMetrics:
    """Diagnostics for the current window."""
    phase: FlightPhase
    slope_per_sec: float
    volatility: float
    imbalance: float
    aligned: bool
    n_points: int
    window_seconds: float


class IntradayFlightClassifier:
    """
    Rolling-window flight classifier for intraday use.

    Typical usage:
        clf = IntradayFlightClassifier(horizon_secs=600)  # last 10 minutes
        ...
        clf.add_sample(time.time(), last_price, bid_size, ask_size)
        metrics = clf.classify()
        print(metrics.phase)
    """

    def __init__(
        self,
        horizon_secs: float = 600.0,
        # thresholds are roughly tuned for SPY; adjust as needed
        strong_slope_cents_per_min: float = 2.0,
        gentle_slope_cents_per_min: float = 0.5,
        low_vol_dollars: float = 0.05,
        high_vol_dollars: float = 0.30,
    ) -> None:
        self.horizon_secs = float(horizon_secs)
        self.samples: Deque[Sample] = deque()

        # Convert slope thresholds into $/sec
        self.strong_slope = (strong_slope_cents_per_min / 100.0) / 60.0
        self.gentle_slope = (gentle_slope_cents_per_min / 100.0) / 60.0

        self.low_vol = float(low_vol_dollars)
        self.high_vol = float(high_vol_dollars)

    # ------------------------------------------------------------------ #
    # Data ingestion
    # ------------------------------------------------------------------ #
    def add_sample(
        self,
        t: Optional[float],
        price: float,
        bid_size: float,
        ask_size: float,
    ) -> None:
        """
        Add a new snapshot.

        Parameters
        ----------
        t : float or None
            Unix timestamp in seconds. If None, uses time.time().
        price : float
            Last traded price.
        bid_size : float
            Aggregate bid size (e.g., sum of top-of-book levels).
        ask_size : float
            Aggregate ask size.
        """
        now = t if t is not None else time.time()
        self.samples.append(Sample(now, float(price), float(bid_size), float(ask_size)))
        self._trim(now)

    def _trim(self, now: float) -> None:
        """Drop samples older than horizon_secs."""
        cutoff = now - self.horizon_secs
        while self.samples and self.samples[0].t < cutoff:
            self.samples.popleft()

    # ------------------------------------------------------------------ #
    # Metrics + classification
    # ------------------------------------------------------------------ #
    def _compute_slope_and_vol(self) -> tuple[float, float, int, float]:
        """
        Simple linear regression slope + standard deviation of price.

        Returns
        -------
        slope_per_sec, vol, n_points, window_seconds
        """
        n = len(self.samples)
        if n < 5:
            return 0.0, 0.0, n, 0.0

        # Use time offset from first point for numerical stability
        t0 = self.samples[0].t
        sum_t = sum_p = sum_tp = sum_tt = 0.0

        for s in self.samples:
            dt = s.t - t0
            p = s.price
            sum_t += dt
            sum_p += p
            sum_tp += dt * p
            sum_tt += dt * dt

        denom = (n * sum_tt - sum_t * sum_t)
        if abs(denom) < 1e-9:
            slope = 0.0
        else:
            slope = (n * sum_tp - sum_t * sum_p) / denom  # $ per second

        mean_p = sum_p / n
        var_p = sum((s.price - mean_p) ** 2 for s in self.samples) / n
        vol = math.sqrt(var_p)

        window_seconds = self.samples[-1].t - self.samples[0].t
        return slope, vol, n, window_seconds

    def _compute_imbalance(self) -> float:
        """
        DOM imbalance in [-1, 1].

        +1  → all size on bid
        -1  → all size on ask
        0   → balanced
        """
        if not self.samples:
            return 0.0

        # Only use the latest sample; you could also average last N.
        s = self.samples[-1]
        total = s.bid_size + s.ask_size
        if total <= 0:
            return 0.0
        return (s.bid_size - s.ask_size) / total

    def classify(self) -> PhaseMetrics:
        """
        Classify the current window into a flight phase.

        Returns
        -------
        PhaseMetrics
            Contains the label plus diagnostic metrics.
        """
        slope, vol, n, window_secs = self._compute_slope_and_vol()

        if n < 5 or window_secs <= 0:
            return PhaseMetrics(
                phase="WARMUP",
                slope_per_sec=slope,
                volatility=vol,
                imbalance=0.0,
                aligned=True,
                n_points=n,
                window_seconds=window_secs,
            )

        imb = self._compute_imbalance()
        abs_slope = abs(slope)

        aligned = ((slope > 0 and imb > 0) or (slope < 0 and imb < 0))

        # Phase rules (rough; tune for your tastes)
        if not aligned and vol > self.low_vol:
            phase: FlightPhase = "TURBULENCE"
        elif abs_slope < self.gentle_slope and vol < self.low_vol:
            phase = "CRUISE"
        elif slope >= self.strong_slope and vol >= self.low_vol:
            phase = "TAKEOFF"
        elif slope >= self.gentle_slope and vol < self.high_vol:
            phase = "CLIMB"
        elif slope <= -self.strong_slope and vol >= self.low_vol:
            phase = "DESCENT"
        elif abs_slope < self.gentle_slope and vol >= self.high_vol:
            phase = "HOLDING"
        else:
            # Unclear direction but not completely flat
            phase = "CLIMB" if slope > 0 else "DESCENT"

        return PhaseMetrics(
            phase=phase,
            slope_per_sec=slope,
            volatility=vol,
            imbalance=imb,
            aligned=aligned,
            n_points=n,
            window_seconds=window_secs,
        )

    # ------------------------------------------------------------------ #
    # Convenience for dashboards
    # ------------------------------------------------------------------ #
    def snapshot(self) -> Dict[str, Any]:
        """
        Lightweight dict for dumping into JSON / websockets.
        """
        m = self.classify()
        return {
            "phase": m.phase,
            "slope_per_sec": m.slope_per_sec,
            "volatility": m.volatility,
            "imbalance": m.imbalance,
            "aligned": m.aligned,
            "n_points": m.n_points,
            "window_seconds": m.window_seconds,
        }


# ---------------------------------------------------------------------- #
# Minimal demo (you can delete this in production)
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import random

    clf = IntradayFlightClassifier(horizon_secs=120)  # last 2 minutes

    now = time.time()
    price = 690.0

    # Simulate a "takeoff": strong upward drift
    for i in range(200):
        now += 0.5
        price += 0.02  # +2 cents every 0.5s → +2.4 $ in 1 min, exaggerated
        bid = random.uniform(500, 900)
        ask = random.uniform(50, 200)
        clf.add_sample(now, price, bid, ask)

    metrics = clf.classify()
    print("Phase:", metrics.phase)
    print("Slope ($/sec):", metrics.slope_per_sec)
    print("Volatility ($):", metrics.volatility)
    print("Imbalance:", metrics.imbalance)
    print("Points:", metrics.n_points, "Window(s):", metrics.window_seconds)
