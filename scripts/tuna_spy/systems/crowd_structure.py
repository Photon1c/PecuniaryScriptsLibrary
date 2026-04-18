# systems/crowd_structure.py
"""
Horizontal axis = emergent crowd structure in the liquidity field (Option A).

Y remains strict price (via particle.price → price_to_y). X is not time or a second
ticker: peers within a small *price* window form a local crowd; cohesion pulls
their screen-x toward the local mean, separation clears overlaps, compression
pulls thick clusters toward the chart center so the ensemble can read as a
compressible blob rather than a vertical smear.
"""
from __future__ import annotations

import random

from config import (
    CROWD_NEIGHBOR_PRICE_SPAN,
    CROWD_COHESION_BASE,
    CROWD_SEPARATION_BASE,
    CROWD_SEPARATION_RADIUS,
    CROWD_COMPRESSION_BASE,
    CROWD_COMPRESSION_MIN_NEIGHBORS,
    CROWD_HORIZONTAL_DAMP,
    CROWD_NOISE,
    CROWD_X_MARGIN_LO_FRAC,
    CROWD_X_MARGIN_HI_FRAC,
    LIVE_CROWD_COHESION_SCALE,
    LIVE_CROWD_COMPRESSION_SCALE,
)


def integrate_crowd_horizontal(particles: list, screen_w: int) -> None:
    alive = [p for p in particles if p.alive]
    n = len(alive)
    if n == 0:
        return

    cohesion = CROWD_COHESION_BASE * LIVE_CROWD_COHESION_SCALE
    separation = CROWD_SEPARATION_BASE
    compression = CROWD_COMPRESSION_BASE * LIVE_CROWD_COMPRESSION_SCALE
    span = CROWD_NEIGHBOR_PRICE_SPAN
    sep_r = CROWD_SEPARATION_RADIUS
    cx = screen_w * 0.5
    lo = screen_w * CROWD_X_MARGIN_LO_FRAC
    hi = screen_w * CROWD_X_MARGIN_HI_FRAC

    mean_x = sum(p.x for p in alive) / n

    for p in alive:
        peers = [q for q in alive if q is not p and abs(q.price - p.price) <= span]
        if peers:
            target_x = sum(q.x for q in peers) / len(peers)
        else:
            target_x = mean_x

        fx = (target_x - p.x) * cohesion

        for q in alive:
            if q is p or abs(q.price - p.price) > span:
                continue
            dx = p.x - q.x
            dist = abs(dx)
            if dist < 1e-4:
                dx = random.choice((-1.0, 1.0))
                dist = 1.0
            if dist < sep_r:
                t = 1.0 - dist / sep_r
                fx += (dx / dist) * separation * (t * t)

        nc = len(peers)
        if nc >= CROWD_COMPRESSION_MIN_NEIGHBORS:
            dens = min(1.55, nc / 10.0)
            fx += (cx - p.x) * compression * dens

        fx += random.gauss(0, CROWD_NOISE)

        p.vx = p.vx * CROWD_HORIZONTAL_DAMP + fx
        p.vx = max(-9.0, min(9.0, p.vx))
        p.x = max(lo, min(hi, p.x + p.vx))
