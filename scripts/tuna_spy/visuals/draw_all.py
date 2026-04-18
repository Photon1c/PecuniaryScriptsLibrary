# visuals/draw_all.py
"""
All rendering in one module for the first pass.
"""
import pygame
import math
import random
from config import (
    WIDTH, HEIGHT,
    SPY_RANGE_LOW, SPY_RANGE_HIGH, SPY_CURRENT,
    STRIKE_OI, CHART_MARGIN,
    C_BG, C_GRID, C_CALL_WELL, C_PUT_WELL, C_BALANCED,
    C_SPINE, C_TEXT, C_ACCENT, C_MAX_PAIN, C_GAMMA_RICH,
    C_PRICE_LINE,
    FONT_SMALL, FONT_MED, FONT_LARGE, FONT_TITLE
)
from systems.liquidity_field import price_to_y, net_acceleration


# ── helpers ──────────────────────────────────────────────────────────────────

def _alpha_surf(size, color, alpha):
    s = pygame.Surface(size, pygame.SRCALPHA)
    s.fill((*color, alpha))
    return s


def _nice_price_ticks(low: float, high: float, max_ticks: int = 12) -> list[float]:
    """Evenly spaced round dollar levels between low and high."""
    span = max(high - low, 1e-6)
    raw = span / max_ticks
    pow10 = 10 ** math.floor(math.log10(max(raw, 0.0001)))
    err = raw / pow10
    if err <= 1.0:
        step = pow10
    elif err <= 2.0:
        step = 2 * pow10
    elif err <= 5.0:
        step = 5 * pow10
    else:
        step = 10 * pow10
    start = math.ceil(low / step) * step
    ticks: list[float] = []
    p = start
    while p <= high + step * 0.001:
        ticks.append(round(p, 6))
        p += step
    return ticks


def _draw_text(surf, font, text, pos, color, align="left"):
    img = font.render(text, True, color)
    x, y = pos
    if align == "right":
        x -= img.get_width()
    elif align == "center":
        x -= img.get_width() // 2
    surf.blit(img, (x, y))


# ── field heatmap (acceleration map) ─────────────────────────────────────────

def draw_field_heatmap(surf: pygame.Surface, field_cache: list, margin: int = CHART_MARGIN):
    """
    Draw a vertical strip on the left showing net gravity direction/magnitude.
    """
    strip_x  = max(36, margin - 32)
    strip_w  = 16
    for (y, price, accel) in field_cache:
        # map accel to color: negative = red (downward), positive = green (upward)
        intensity = min(abs(accel) / 0.8, 1.0)
        if accel > 0:   # price pulled upward
            color = (int(20 + 40 * intensity), int(150 + 70 * intensity), int(80 + 60 * intensity))
        else:           # price pulled downward
            color = (int(150 + 80 * intensity), int(30 + 40 * intensity), int(40 + 40 * intensity))
        pygame.draw.line(surf, color, (strip_x, y), (strip_x + strip_w, y))


# ── grid and price axis ───────────────────────────────────────────────────────

def draw_price_ruler(surf: pygame.Surface, font_m, margin: int = CHART_MARGIN):
    """Left-side dollar scale + tick marks (readable price axis)."""
    lo, hi = SPY_RANGE_LOW, SPY_RANGE_HIGH
    ticks = _nice_price_ticks(lo, hi)
    axis_x = margin - 4
    pygame.draw.line(
        surf,
        (*C_GRID, 200),
        (axis_x, margin),
        (axis_x, HEIGHT - margin),
        2,
    )
    for p in ticks:
        y = int(price_to_y(p, HEIGHT, margin))
        if y < margin - 2 or y > HEIGHT - margin + 2:
            continue
        pygame.draw.line(surf, (*C_GRID, 140), (axis_x - 6, y), (axis_x + 2, y), 1)
        lbl = f"{p:.0f}"
        _draw_text(surf, font_m, lbl, (6, y - 11), C_TEXT)


def draw_price_axis(surf: pygame.Surface, font_s, font_m, margin: int = CHART_MARGIN):
    """Horizontal gridlines at each OI strike; labels skip when strikes are dense."""
    grid_x0 = margin + 8
    label_x = margin + 10
    min_label_dy = 15
    prev_lbl_y = -1e9
    for strike in sorted(STRIKE_OI.keys()):
        y = int(price_to_y(strike, HEIGHT, margin))
        call_oi, put_oi = STRIKE_OI[strike]
        net = call_oi - put_oi

        line_alpha = 35 + min(abs(net), 60)
        pygame.draw.line(surf, C_GRID, (grid_x0, y), (WIDTH - 10, y))

        if abs(y - prev_lbl_y) >= min_label_dy:
            _draw_text(surf, font_m, f"{strike}", (label_x, y - 10), C_ACCENT)
            prev_lbl_y = y
        else:
            _draw_text(surf, font_s, "·", (label_x, y - 6), C_GRID)


# ── OI wells (the gravity wells themselves) ───────────────────────────────────

def draw_oi_wells(surf: pygame.Surface, font_s, max_pain: float, margin: int = CHART_MARGIN):
    """
    For each strike: draw proportional call (right) and put (left) bars,
    plus a glowing halo scaled to total OI.
    """
    mid_x    = WIDTH // 2
    bar_scale = 1.8   # px per 1k contracts

    max_total = max(c + p for c, p in STRIKE_OI.values())

    for strike, (call_oi, put_oi) in STRIKE_OI.items():
        y       = int(price_to_y(strike, HEIGHT, margin))
        total   = call_oi + put_oi
        net     = call_oi - put_oi
        balance = abs(net) / max(total, 1)

        # Well color: balanced → gold, call-heavy → teal, put-heavy → red
        if balance < 0.15:
            well_color = C_BALANCED
        elif net > 0:
            well_color = C_CALL_WELL
        else:
            well_color = C_PUT_WELL

        # Max pain highlight
        is_max_pain = abs(strike - max_pain) < 0.5
        if is_max_pain:
            well_color = C_MAX_PAIN

        # ATM gamma highlight
        is_atm = abs(strike - SPY_CURRENT) < 3
        if is_atm:
            well_color = C_GAMMA_RICH

        # Halo glow (larger for bigger OI)
        halo_r = int(4 + (total / max_total) * 28)
        halo_surf = pygame.Surface((halo_r * 2 + 2, halo_r * 2 + 2), pygame.SRCALPHA)
        for r in range(halo_r, 0, -1):
            alpha = int(8 + (1 - r / halo_r) * 40)
            pygame.draw.circle(halo_surf, (*well_color, alpha),
                               (halo_r + 1, halo_r + 1), r)
        surf.blit(halo_surf, (mid_x - halo_r - 1, y - halo_r - 1))

        # Call bar (right of center)
        call_w = int(call_oi * bar_scale)
        call_rect = pygame.Rect(mid_x + 2, y - 3, call_w, 6)
        pygame.draw.rect(surf, (*C_CALL_WELL, 160), call_rect, border_radius=2)

        # Put bar (left of center, mirrored)
        put_w = int(put_oi * bar_scale)
        put_rect = pygame.Rect(mid_x - 2 - put_w, y - 3, put_w, 6)
        pygame.draw.rect(surf, (*C_PUT_WELL, 160), put_rect, border_radius=2)

        # OI label on right
        label = f"C:{call_oi}k  P:{put_oi}k"
        _draw_text(surf, font_s, label, (mid_x + call_w + 8, y - 6), (*well_color, 200))

        # Max pain label
        if is_max_pain:
            _draw_text(surf, font_s, "MAX PAIN", (mid_x + call_w + 8, y + 4),
                       (*C_MAX_PAIN, 220))


# ── particle trails and dots ──────────────────────────────────────────────────

def draw_particles(surf: pygame.Surface, particles: list):
    for p in particles:
        if not p.alive:
            continue
        col = p.color()
        # Trail
        if len(p.trail) > 1:
            for i in range(1, len(p.trail)):
                alpha = int(30 + (i / len(p.trail)) * 80)
                a, b  = p.trail[i-1], p.trail[i]
                tr_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
                pygame.draw.line(
                    surf,
                    (*col, alpha),
                    (int(a[0]), int(a[1])),
                    (int(b[0]), int(b[1])),
                    1
                )
        # Dot
        pygame.draw.circle(surf, col, (int(p.x), int(p.y)), 3)


# ── price spine (ensemble centroid ribbon) ────────────────────────────────────

def draw_price_spine(surf: pygame.Surface, spine_history: list):
    if len(spine_history) < 2:
        return
    for i in range(1, len(spine_history)):
        ax, ay = spine_history[i-1]
        bx, by = spine_history[i]
        alpha = int(60 + (i / len(spine_history)) * 180)
        pygame.draw.line(surf, (*C_SPINE, min(alpha, 255)),
                         (int(ax), int(ay)), (int(bx), int(by)), 2)
    # Current dot
    cx, cy = spine_history[-1]
    pygame.draw.circle(surf, C_SPINE, (int(cx), int(cy)), 5)


# ── current price line ────────────────────────────────────────────────────────

def draw_current_price_line(surf: pygame.Surface, price: float, font_m, margin: int = CHART_MARGIN):
    y = int(price_to_y(price, HEIGHT, margin))
    grid_x0 = margin + 8
    pygame.draw.line(surf, (*C_PRICE_LINE, 200), (grid_x0, y), (WIDTH - 10, y), 2)
    _draw_text(surf, font_m, f"SPY  {price:.2f}", (grid_x0 + 4, y - 18),
               C_PRICE_LINE)


# ── HUD ───────────────────────────────────────────────────────────────────────

def draw_hud(surf: pygame.Surface, fonts: dict, metrics: dict,
             phase_mgr, field_cache):
    font_s = fonts["small"]
    font_m = fonts["med"]
    font_l = fonts["large"]
    font_t = fonts["title"]

    # ── Title bar ────────────────────────────────────────────────────────
    pygame.draw.rect(surf, (10, 14, 30), (0, 0, WIDTH, 44))
    _draw_text(surf, font_t, "SPY  OI  GRAVITY  WELLS",
               (WIDTH // 2, 10), C_ACCENT, align="center")

    # ── Phase badge ──────────────────────────────────────────────────────
    phase_col = phase_mgr.phase_color()
    badge_text = phase_mgr.phase.upper().replace("_", " ")
    _draw_text(surf, font_l, badge_text, (WIDTH - 20, 10), phase_col, align="right")

    # ── Left panel ───────────────────────────────────────────────────────
    lx, ly = 78, 52
    dy = 18

    mean_p  = metrics.get("mean_price",       0)
    std_p   = metrics.get("std_price",        0)
    cf      = metrics.get("cluster_fraction", 0)
    ds      = metrics.get("dominant_strike",  0)
    mp      = phase_mgr.max_pain

    rows = [
        ("Mean Price",  f"{mean_p:.2f}"),
        ("Std Dev",     f"±{std_p:.2f}"),
        ("Cluster %",   f"{cf*100:.1f}%"),
        ("Pin Strike",  f"{ds}"),
        ("Max Pain",    f"{mp:.0f}"),
        ("Tick",        f"{phase_mgr.tick}"),
    ]
    for label, val in rows:
        _draw_text(surf, font_s, label, (lx, ly),      C_TEXT)
        _draw_text(surf, font_s, val,   (lx + 82, ly), C_ACCENT)
        ly += dy

    ly += 6
    _draw_text(surf, font_s, "Y = price", (lx, ly), C_ACCENT)
    ly += dy
    _draw_text(surf, font_s, "X = crowd (local)", (lx, ly), C_ACCENT)

    # ── Legend ────────────────────────────────────────────────────────────
    lx2, ly2 = WIDTH - 200, 52
    legend = [
        (C_CALL_WELL, "Call OI"),
        (C_PUT_WELL,  "Put OI"),
        (C_BALANCED,  "Balanced strike"),
        (C_MAX_PAIN,  "Max Pain"),
        (C_GAMMA_RICH,"ATM Gamma-rich"),
    ]
    for col, lbl in legend:
        pygame.draw.rect(surf, col, (lx2, ly2 + 3, 10, 10), border_radius=2)
        _draw_text(surf, font_s, lbl, (lx2 + 15, ly2), C_TEXT)
        ly2 += 18

    # ── Event log ─────────────────────────────────────────────────────────
    events = phase_mgr.recent_events(4)
    ey = HEIGHT - 14 - len(events) * 15
    for tick, msg in events:
        _draw_text(surf, font_s, f"t{tick:05d}  {msg}", (82, ey), C_ACCENT)
        ey += 15

    # ── Controls hint ─────────────────────────────────────────────────────
    hints = "[R] Reset   [N] Noise+   [E] Expiry   [Q] Quit"
    _draw_text(surf, font_s, hints, (WIDTH // 2, HEIGHT - 14), C_TEXT, align="center")
