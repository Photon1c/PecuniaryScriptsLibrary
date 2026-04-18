# systems/phase_manager.py
"""
Manages simulation phases and triggers events like gamma expiry.
"""
from config import CLUSTER_THRESHOLD, GAMMA_EXPIRY_TICKS, SPY_CURRENT, STRIKE_OI
from systems.liquidity_field import compute_max_pain


class PhaseManager:
    PHASES = ["dispersed", "clustering", "pinned", "expiry_crush", "rebalance"]

    def __init__(self):
        self.phase       = "dispersed"
        self.tick        = 0
        self.max_pain    = compute_max_pain(STRIKE_OI)
        self.events      = []   # list of (tick, description) log entries
        self.noise_scale = 1.0
        self.expiry_fired = False

    def update(self, metrics: dict):
        self.tick += 1
        cf = metrics.get("cluster_fraction", 0)
        ds = metrics.get("dominant_strike", SPY_CURRENT)
        mp = self.max_pain

        # Phase transitions
        if self.phase == "dispersed":
            if cf > 0.30:
                self._transition("clustering")

        elif self.phase == "clustering":
            if cf > CLUSTER_THRESHOLD:
                self._transition("pinned")
            elif cf < 0.15:
                self._transition("dispersed")

        elif self.phase == "pinned":
            if cf < 0.35:
                self._transition("clustering")
            # Gamma squeeze if near max pain and tick approaches expiry
            if abs(ds - mp) < 5 and self.tick > GAMMA_EXPIRY_TICKS // 2:
                self._transition("expiry_crush")

        elif self.phase == "expiry_crush":
            if not self.expiry_fired and self.tick > GAMMA_EXPIRY_TICKS:
                self._fire_expiry()
            if self.tick > GAMMA_EXPIRY_TICKS + 300:
                self._transition("rebalance")

        elif self.phase == "rebalance":
            if self.tick > GAMMA_EXPIRY_TICKS + 900:
                self._reset()

        # Noise scaling per phase
        noise_map = {
            "dispersed":    1.4,
            "clustering":   0.9,
            "pinned":       0.5,
            "expiry_crush": 2.2,
            "rebalance":    1.1,
        }
        self.noise_scale = noise_map.get(self.phase, 1.0)

    def _transition(self, new_phase: str):
        self.events.append((self.tick, f"→ {new_phase.upper()}"))
        self.phase = new_phase

    def _fire_expiry(self):
        self.expiry_fired = True
        self.events.append((self.tick, "⚡ GAMMA EXPIRY — OI resets"))

    def _reset(self):
        self.tick         = 0
        self.expiry_fired = False
        self.phase        = "dispersed"
        self.events.clear()

    def phase_color(self) -> tuple:
        colors = {
            "dispersed":    (100, 120, 200),
            "clustering":   (80,  200, 160),
            "pinned":       (255, 220,  60),
            "expiry_crush": (255,  80,  60),
            "rebalance":    (160, 100, 255),
        }
        return colors.get(self.phase, (200, 200, 200))

    def recent_events(self, n: int = 5) -> list:
        return self.events[-n:]
