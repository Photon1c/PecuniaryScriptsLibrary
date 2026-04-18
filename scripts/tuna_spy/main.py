# main.py — SPY OI Gravity Well Simulator
"""
Run with:  python main.py
           python main.py --live

Controls:
  R  — reset particles
  N  — inject noise spike (simulate news shock)
  E  — force gamma expiry event
  +  — increase noise
  -  — decrease noise
  Q  — quit
"""
import argparse
import sys
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

# --live patches config before any submodule imports `from config import …`
_parser = argparse.ArgumentParser(description="SPY OI Gravity Well Simulator")
_parser.add_argument(
    "--live",
    action="store_true",
    help="Load newest CSV per folder whose filename contains --ticker (default SPY)",
)
_parser.add_argument(
    "--stocks-dir",
    default=r"F:\inputs\stocks",
    metavar="DIR",
    help="Folder with stock historical CSVs (default: F:\\inputs\\stocks)",
)
_parser.add_argument(
    "--options-dir",
    default=r"F:\inputs\options\log",
    metavar="DIR",
    help="Folder with options chain CSV logs (default: F:\\inputs\\options\\log)",
)
_parser.add_argument(
    "--ticker",
    default="SPY",
    metavar="SYM",
    help="Only use CSVs whose filename contains this symbol (default: SPY)",
)
_CLI = _parser.parse_args()
if _CLI.live:
    from live_data import apply_live_config
    apply_live_config(_CLI.stocks_dir, _CLI.options_dir, ticker=_CLI.ticker)

import pygame
import random

from config import (
    WIDTH, HEIGHT, FPS, TITLE,
    NUM_PARTICLES, SPY_CURRENT,
    SPY_RANGE_LOW, SPY_RANGE_HIGH,
    FONT_SMALL, FONT_MED, FONT_LARGE, FONT_TITLE,
)
from entities.particle import PriceParticle
from systems.liquidity_field import build_field_cache, price_to_y
from systems.metrics import cluster_metrics, gamma_exposure, field_pressure
from systems.phase_manager import PhaseManager
from systems.crowd_structure import integrate_crowd_horizontal
from visuals.draw_all import (
    draw_field_heatmap, draw_price_ruler, draw_price_axis, draw_oi_wells,
    draw_particles, draw_price_spine, draw_current_price_line,
    draw_hud,
)


def make_fonts():
    pygame.font.init()
    try:
        mono = pygame.font.SysFont("Courier New", FONT_SMALL)
        mono_m = pygame.font.SysFont("Courier New", FONT_MED)
        mono_l = pygame.font.SysFont("Courier New", FONT_LARGE)
        mono_t = pygame.font.SysFont("Courier New", FONT_TITLE, bold=True)
    except Exception:
        mono   = pygame.font.Font(None, FONT_SMALL + 2)
        mono_m = pygame.font.Font(None, FONT_MED  + 2)
        mono_l = pygame.font.Font(None, FONT_LARGE + 2)
        mono_t = pygame.font.Font(None, FONT_TITLE + 4)
    return {"small": mono, "med": mono_m, "large": mono_l, "title": mono_t}


def spawn_particles(n: int) -> list:
    return [PriceParticle(HEIGHT, WIDTH) for _ in range(n)]


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(TITLE)
    clock  = pygame.time.Clock()

    fonts   = make_fonts()
    phase   = PhaseManager()
    field   = build_field_cache(HEIGHT)
    parts   = spawn_particles(NUM_PARTICLES)

    spine_history = []
    spine_max     = 300
    noise_boost   = 1.0

    # Pre-render static OI well layer (strikes don't move)
    static_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    draw_price_ruler(static_surf, fonts["med"])
    draw_price_axis(static_surf, fonts["small"], fonts["med"])
    draw_oi_wells(static_surf, fonts["small"], phase.max_pain)
    draw_field_heatmap(static_surf, field)

    running = True
    while running:
        # ── Events ───────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    parts = spawn_particles(NUM_PARTICLES)
                    spine_history.clear()
                    phase.__init__()
                elif event.key == pygame.K_n:
                    noise_boost = 4.0
                    phase.events.append((phase.tick, "⚡ NOISE SHOCK injected"))
                elif event.key == pygame.K_e:
                    phase.phase = "expiry_crush"
                    phase.events.append((phase.tick, "⚡ MANUAL EXPIRY triggered"))
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    noise_boost = min(noise_boost + 0.5, 8.0)
                elif event.key == pygame.K_MINUS:
                    noise_boost = max(noise_boost - 0.5, 0.2)

        # ── Update ───────────────────────────────────────────────────────
        noise_boost = max(noise_boost * 0.995, 1.0)   # decay boost

        for p in parts:
            p.update(noise_scale=noise_boost * phase.noise_scale)
            # Respawn if out of range
            if p.price <= SPY_RANGE_LOW + 1 or p.price >= SPY_RANGE_HIGH - 1:
                p.reset_near_current()

        integrate_crowd_horizontal(parts, WIDTH)
        for p in parts:
            p.sync_trail()

        metrics = cluster_metrics(parts)
        phase.update(metrics)

        # Spine (centroid of alive particles)
        alive = [p for p in parts if p.alive]
        if alive:
            cx = sum(p.x for p in alive) / len(alive)
            cy = sum(p.y for p in alive) / len(alive)
            spine_history.append((cx, cy))
            if len(spine_history) > spine_max:
                spine_history.pop(0)

        # ── Draw ─────────────────────────────────────────────────────────
        screen.fill((6, 8, 18))
        screen.blit(static_surf, (0, 0))   # static wells layer

        draw_price_spine(screen, spine_history)
        draw_particles(screen, parts)

        # Current ensemble mean price line
        mean_p = metrics.get("mean_price", SPY_CURRENT)
        draw_current_price_line(screen, mean_p, fonts["med"])

        draw_hud(screen, fonts, metrics, phase, field)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
