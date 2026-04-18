# entities/particle.py
"""
PriceParticle — a single probe of the price field.

Each particle represents a possible price trajectory.
The ensemble of particles shows the probability distribution
that emerges from the OI gravity wells.
"""
import random
from config import (
    SPY_CURRENT, SPY_RANGE_LOW, SPY_RANGE_HIGH,
    PARTICLE_MASS_MIN, PARTICLE_MASS_MAX,
    DAMPING, NOISE_STRENGTH,
    C_PARTICLE, C_PARTICLE_HOT,
)
from systems.liquidity_field import price_to_y, net_acceleration
from utils.vectors import clamp, lerp


class PriceParticle:
    def __init__(self, screen_height: int, screen_width: int, spread: float = 8.0):
        self.H = screen_height
        self.W = screen_width

        # Scatter particles around current price at start
        start_price = SPY_CURRENT + random.gauss(0, spread)
        start_price = clamp(start_price, SPY_RANGE_LOW + 2, SPY_RANGE_HIGH - 2)

        self.price = start_price
        self.vel   = random.gauss(0, 0.3)   # price units/tick
        self.mass  = random.uniform(PARTICLE_MASS_MIN, PARTICLE_MASS_MAX)

        # Horizontal position starts scattered; crowd_structure.py evolves X.
        self.x     = random.uniform(self.W * 0.12, self.W * 0.88)
        self.vx    = 0.0

        self.heat  = 0.0    # 0..1 — how "pinned / active" this particle is
        self.trail = []     # last N (x, y) positions
        self.trail_max = 30
        self.alive = True

    @property
    def y(self):
        return price_to_y(self.price, self.H)

    def update(self, noise_scale: float = 1.0):
        if not self.alive:
            return

        # Gravity from OI field
        accel = net_acceleration(self.price) / self.mass

        # Brownian noise
        noise = random.gauss(0, NOISE_STRENGTH * noise_scale)

        self.vel = self.vel * DAMPING + accel + noise
        self.price += self.vel
        self.price  = clamp(self.price, SPY_RANGE_LOW, SPY_RANGE_HIGH)

        # Update heat based on velocity
        self.heat = clamp(self.heat * 0.97 + abs(self.vel) * 0.3, 0.0, 1.0)

    def sync_trail(self):
        """Call after horizontal crowd integration so trails use final x,y."""
        if not self.alive:
            return
        self.trail.append((self.x, self.y))
        if len(self.trail) > self.trail_max:
            self.trail.pop(0)

    def color(self):
        r = int(lerp(C_PARTICLE[0], C_PARTICLE_HOT[0], self.heat))
        g = int(lerp(C_PARTICLE[1], C_PARTICLE_HOT[1], self.heat))
        b = int(lerp(C_PARTICLE[2], C_PARTICLE_HOT[2], self.heat))
        return (r, g, b)

    def reset_near_current(self, spread: float = 5.0):
        """Respawn near current price after going out of range."""
        self.price = clamp(SPY_CURRENT + random.gauss(0, spread),
                           SPY_RANGE_LOW + 1, SPY_RANGE_HIGH - 1)
        self.vel   = random.gauss(0, 0.2)
        self.vx    = 0.0
        self.x     = random.uniform(self.W * 0.15, self.W * 0.85)
        self.trail.clear()
