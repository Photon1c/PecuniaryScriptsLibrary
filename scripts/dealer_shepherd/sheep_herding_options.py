# Sheep Herding - Options
# Use data_loader.py (pending)
# Controls:
#   Space = toggle breach (put/call)   R = reset   +/- = IV down/up   ESC = quit
#   T = toggle ticker (SPY/QQQ/etc)     P = print price zones
import argparse
import pygame as pg, random, math, sys
import csv
import os
import sys
from enum import Enum
from datetime import datetime

# Import data_loader - handle both relative and absolute imports
try:
    from data_loader import get_latest_price, load_stock_data
except ImportError:
    # If running from parent directory, add current dir to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from data_loader import get_latest_price, load_stock_data

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - Adjust these values to customize the simulation
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
W, H = 960, 600  # Window width and height

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
N = 80  # Number of sheep in the flock
TICKER_DEFAULT = "SPY"  # Default ticker to load
TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "TSLA"]  # Available tickers (press T to cycle)

# ─────────────────────────────────────────────────────────────────────────────
# GAMMA ZONE SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
PEN_WIDTH = 280  # Width of gamma-neutral zone (pen)
PEN_HEIGHT = 180  # Height of gamma-neutral zone (pen)
STRIKE_SPACING = 5.0  # Default strike spacing for gamma calculations ($5 for SPY)

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
# Dog (Dealer) Physics
DOG_SPD_BASE = 2.2  # Base speed of dealer dog
DOG_TARGET_SMOOTH = 0.18  # EMA smoothing for target (reduces jitter)
DOG_DAMP = 0.28  # Velocity damping coefficient
DOG_A_MAX = 0.90  # Maximum acceleration (pixels/frame^2)
DOG_HERDING_PUSH = 0.20  # Strength of dog's repulsion when herding sheep
DOG_HERDING_RADIUS = 120  # Distance at which dog affects sheep

# Sheep (Retail) Physics
SHEEP_SPD = 1.2  # Base speed of retail sheep
SHEEP_PEN_PUSH = 0.03  # Push toward pen when dog is herding (greener pastures effect)

# ─────────────────────────────────────────────────────────────────────────────
# FLOCKING BEHAVIOR SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
FLOCK_SEPARATION = 0.04  # Avoid crowding neighbors (reduced to prevent tight clustering)
FLOCK_ALIGNMENT = 0.02   # Steer towards average heading (reduced for more directional movement)
FLOCK_COHESION = 0.01    # Steer towards average position (reduced to prevent circling)
FLOCK_RADIUS = 50        # Neighbor detection radius for flocking

# ─────────────────────────────────────────────────────────────────────────────
# GRAZING BEHAVIOR SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
GRAZING_ATTRACTION = 0.15  # Attraction strength to spot price when far away
GRAZING_RADIUS = 100      # Radius of grazing area around spot price
GRAZING_SETTLE_RADIUS = 40  # Within this radius, sheep slow down and settle
GRAZING_WANDER = 0.03     # Random wander while grazing (reduced for focus)
GRAZING_SETTLE_DAMP = 0.85  # Velocity damping when in settle zone (lower = more settling)

# ─────────────────────────────────────────────────────────────────────────────
# BOUNDARY SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
BOUNDARY_MARGIN = 30  # Distance from edge where repulsion starts
BOUNDARY_FORCE = 0.08  # Strength of boundary repulsion

# ─────────────────────────────────────────────────────────────────────────────
# BREACH SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
BREACH_MAX_DURATION = 300  # Breach duration in frames (~5 seconds at 60fps)
BREACH_RECOVERY_SPEED = 0.02  # Speed at which price returns to flip (0.0-1.0)
BREACH_MAGNITUDE = 1.5  # How many strikes the price moves during breach

# ─────────────────────────────────────────────────────────────────────────────
# IV (IMPLIED VOLATILITY) SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
IV_DEFAULT = 10  # Default IV percentage
IV_MIN = 0  # Minimum IV percentage
IV_MAX = 80  # Maximum IV percentage
IV_STEP = 2  # IV change per +/- keypress

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
BACKTEST_ENABLED = True  # Enable historical backtest mode
BACKTEST_STEP_DAYS = 1  # Days to step forward/backward per arrow key
BACKTEST_AUTO_PLAY = False  # Auto-play through history
BACKTEST_AUTO_PLAY_SPEED = 1  # Frames per step (lower = faster)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING (opt-in)
# ─────────────────────────────────────────────────────────────────────────────
ENABLE_LOGGING = True  # default off
LOG_FILE_PATH = "logs/sheep_herding_options_log.csv"
LOG_EVERY_N_FRAMES = 5  # only log every Nth frame to reduce noise

# ─────────────────────────────────────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────────────────────────────────────
COL_BG = (18, 18, 22)  # Background color
COL_PEN = (74, 74, 82)  # Gamma-neutral pen color
COL_PEN_BORDER = (120, 120, 140)  # Pen border color
COL_DOG = (30, 180, 255)  # Dealer dog color
COL_SHEEP = (240, 240, 240)  # Retail sheep color
COL_SHEEP_IN_PEN = (255, 255, 180)  # Sheep color when in pen
COL_PUT = (210, 60, 60)  # Put breach overlay color
COL_CALL = (60, 200, 90)  # Call breach overlay color
COL_NEG_GAMMA = (180, 60, 60)  # Negative gamma zone color
COL_POS_GAMMA = (60, 150, 80)  # Positive gamma zone color
COL_TEXT = (220, 220, 230)  # Text color

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL CONSTANTS (Don't modify unless you know what you're doing)
# ─────────────────────────────────────────────────────────────────────────────
BREACH_NONE, BREACH_PUT, BREACH_CALL = 0, 1, 2  # Breach state constants


class RegimeState(Enum):
    """Scaffold for Markov / flight-envelope regime. Not yet used for behavior."""
    RANGE_BOUND = "range_bound"
    BREACH_UP = "breach_up"      # call-side rupture
    BREACH_DOWN = "breach_down"  # put-side rupture


# Initialize pen rectangle (calculated from settings above)
PEN = pg.Rect(W//2 - PEN_WIDTH//2, H//2 - PEN_HEIGHT//2, PEN_WIDTH, PEN_HEIGHT)

# Logging handle (set by init_logger when ENABLE_LOGGING)
_log_file = None
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Price zone calculations
def calculate_gamma_zones(spot_price, strike_spacing=None):
    """Calculate gamma regime zones based on spot price"""
    if strike_spacing is None:
        strike_spacing = STRIKE_SPACING
    """
    Calculate gamma regime zones based on spot price.
    Returns: (gamma_flip, put_wall, call_wall, neg_gamma_zone, pos_gamma_zone)
    """
    # Estimate gamma flip near spot (typically at-the-money)
    gamma_flip = round(spot_price / strike_spacing) * strike_spacing
    
    # Estimate walls (typically ±2-3 strikes from flip)
    put_wall = gamma_flip - (strike_spacing * 2.5)
    call_wall = gamma_flip + (strike_spacing * 2.5)
    
    # Negative gamma zone: below flip (dealers short gamma, price accelerates)
    neg_gamma_zone = (put_wall - strike_spacing * 3, gamma_flip)
    
    # Positive gamma zone: above flip (dealers long gamma, price pinned)
    pos_gamma_zone = (gamma_flip, call_wall + strike_spacing * 3)
    
    return {
        'spot': spot_price,
        'gamma_flip': gamma_flip,
        'put_wall': put_wall,
        'call_wall': call_wall,
        'neg_gamma_range': neg_gamma_zone,
        'pos_gamma_range': pos_gamma_zone,
        'strike_spacing': strike_spacing
    }

def price_to_screen_x(price, zones, screen_width):
    """Convert price to screen X coordinate"""
    price_range = zones['pos_gamma_range'][1] - zones['neg_gamma_range'][0]
    price_min = zones['neg_gamma_range'][0]
    normalized = (price - price_min) / price_range
    return int(normalized * screen_width)

def screen_x_to_price(x, zones, screen_width):
    """Convert screen X coordinate to price"""
    price_range = zones['pos_gamma_range'][1] - zones['neg_gamma_range'][0]
    price_min = zones['neg_gamma_range'][0]
    normalized = x / screen_width
    return price_min + (normalized * price_range)

def in_pen(x, y):
    return PEN.collidepoint(x, y)


def compute_sheep_centroid(flock):
    """Centroid (x, y) of all sheep. Used for logging and analysis."""
    if not flock:
        return (0.0, 0.0)
    x = sum(s.x for s in flock) / len(flock)
    y = sum(s.y for s in flock) / len(flock)
    return (x, y)


def init_logger():
    """Create logs dir, open CSV in append mode, write header only if new/empty.
    Log path is resolved relative to script dir so logs always go to
    .../metascripts/quickscripts/logs/sheep_herding_log.csv regardless of cwd."""
    global _log_file
    if not ENABLE_LOGGING:
        return
    log_path = os.path.normpath(os.path.join(_SCRIPT_DIR, LOG_FILE_PATH))
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    existed = os.path.isfile(log_path)
    _log_file = open(log_path, "a", newline="", encoding="utf-8")
    need_header = not existed
    if existed and os.path.getsize(log_path) == 0:
        need_header = True
    if need_header:
        w = csv.writer(_log_file)
        w.writerow([
            "datetime", "timestamp", "ticker", "spot", "gamma_flip", "put_wall", "call_wall",
            "iv", "breach_state", "n_sheep_total", "n_sheep_grazing_near_spot",
            "n_sheep_in_pen", "dog_x", "dog_y", "sheep_centroid_x", "sheep_centroid_y",
            "regime_state"
        ])
        _log_file.flush()
    print(f"Logging to {log_path}")


def _csv_sanitize(v):
    """Coerce value for CSV; NaN/None/Inf → 0 so downstream parsers stay happy."""
    if v is None:
        return 0
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return 0
    if isinstance(v, (int, float)):
        return round(v, 6) if isinstance(v, float) else v
    return v


def log_frame(frame_count, ticker, zones, iv, breach, flock, dog, n_grazing, n_in_pen, regime_state):
    """Append one CSV row. Call only when ENABLE_LOGGING and frame_count % LOG_EVERY_N_FRAMES == 0."""
    global _log_file
    if not ENABLE_LOGGING or _log_file is None:
        return
    cx, cy = compute_sheep_centroid(flock)
    breach_str = ("NONE", "PUT", "CALL")[breach]
    w = csv.writer(_log_file)
    w.writerow([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        _csv_sanitize(frame_count), ticker or "",
        _csv_sanitize(zones.get("spot")), _csv_sanitize(zones.get("gamma_flip")),
        _csv_sanitize(zones.get("put_wall")), _csv_sanitize(zones.get("call_wall")),
        _csv_sanitize(iv), breach_str, len(flock), n_grazing, n_in_pen,
        _csv_sanitize(dog.x), _csv_sanitize(dog.y), _csv_sanitize(cx), _csv_sanitize(cy),
        regime_state.value if regime_state else ""
    ])
    _log_file.flush()


def get_historical_price(df, index):
    """Get price and date from historical data by index"""
    if df is None or len(df) == 0 or index < 0 or index >= len(df):
        return None, None
    
    row = df.iloc[index]
    
    # Extract price
    if "Close/Last" in df.columns:
        price_raw = row["Close/Last"]
        price = float(str(price_raw).replace("$", "").replace(",", "").strip())
    elif "Close" in df.columns:
        price_raw = row["Close"]
        price = float(str(price_raw).replace("$", "").replace(",", "").strip()) if isinstance(price_raw, str) else float(price_raw)
    else:
        return None, None
    
    # Extract date
    date = row["Date"] if "Date" in df.columns else None
    
    return price, date

def draw_backtest_slider(screen, current_index, total_records, date_str):
    """Draw backtest slider and date indicator"""
    slider_y = 5
    slider_height = 20
    slider_width = W - 40
    slider_x = 20
    
    # Background
    pg.draw.rect(screen, (30, 30, 35), 
                 (slider_x, slider_y, slider_width, slider_height))
    pg.draw.rect(screen, (60, 60, 70), 
                 (slider_x, slider_y, slider_width, slider_height), 2)
    
    # Slider position
    if total_records > 0:
        slider_pos = slider_x + int((current_index / max(1, total_records - 1)) * slider_width)
        # Draw slider handle
        pg.draw.circle(screen, (100, 150, 255), (slider_pos, slider_y + slider_height // 2), 8)
        pg.draw.line(screen, (100, 150, 255), 
                    (slider_x, slider_y + slider_height // 2), 
                    (slider_pos, slider_y + slider_height // 2), 2)
    
    # Date and position text
    font = pg.font.SysFont(None, 16)
    date_text = f"{date_str}" if date_str else f"Record {current_index + 1}/{total_records}"
    text_surface = font.render(date_text, True, (220, 220, 230))
    screen.blit(text_surface, (slider_x + 5, slider_y + slider_height + 2))
    
    # Position indicator
    pos_text = f"{current_index + 1}/{total_records}"
    pos_surface = font.render(pos_text, True, (180, 180, 200))
    screen.blit(pos_surface, (slider_x + slider_width - pos_surface.get_width() - 5, slider_y + slider_height + 2))


class Sheep:
    def __init__(s):
        s.x = random.uniform(80, W - 80)
        s.y = random.uniform(80, H - 80)
        s.vx = random.uniform(-0.5, 0.5)
        s.vy = random.uniform(-0.5, 0.5)
        s.grazing_time = 0  # Time spent in grazing area

    def step(s, flock, spot_x, spot_y, dog_pos, zones):
        """Update sheep with flocking and grazing behavior"""
        # Find neighbors for flocking
        neighbors = [sheep for sheep in flock if sheep != s and 
                     math.hypot(sheep.x - s.x, sheep.y - s.y) < FLOCK_RADIUS]
        
        # 1. SEPARATION: Avoid crowding neighbors
        sep_x, sep_y = 0.0, 0.0
        for neighbor in neighbors:
            dx = s.x - neighbor.x
            dy = s.y - neighbor.y
            dist = math.hypot(dx, dy) + 1e-6
            if dist < 30:  # Too close
                sep_x += (dx / dist) / dist
                sep_y += (dy / dist) / dist
        s.vx += sep_x * FLOCK_SEPARATION
        s.vy += sep_y * FLOCK_SEPARATION
        
        # 2. ALIGNMENT: Steer towards average heading of neighbors
        if neighbors:
            avg_vx = sum(n.vx for n in neighbors) / len(neighbors)
            avg_vy = sum(n.vy for n in neighbors) / len(neighbors)
            s.vx += (avg_vx - s.vx) * FLOCK_ALIGNMENT
            s.vy += (avg_vy - s.vy) * FLOCK_ALIGNMENT
        
        # 3. COHESION: Steer towards average position of neighbors
        if neighbors:
            avg_x = sum(n.x for n in neighbors) / len(neighbors)
            avg_y = sum(n.y for n in neighbors) / len(neighbors)
            s.vx += (avg_x - s.x) * FLOCK_COHESION
            s.vy += (avg_y - s.y) * FLOCK_COHESION
        
        # 4. GRAZING: Focused directional movement toward spot price
        dist_to_spot = math.hypot(s.x - spot_x, s.y - spot_y)
        
        if dist_to_spot < GRAZING_RADIUS:
            # Within grazing radius - move toward spot
            if dist_to_spot > GRAZING_SETTLE_RADIUS:
                # Far from spot: strong directional attraction
                attraction = GRAZING_ATTRACTION * (1.0 - (dist_to_spot - GRAZING_SETTLE_RADIUS) / (GRAZING_RADIUS - GRAZING_SETTLE_RADIUS))
                dir_x = (spot_x - s.x) / (dist_to_spot + 1e-6)
                dir_y = (spot_y - s.y) / (dist_to_spot + 1e-6)
                s.vx += dir_x * attraction
                s.vy += dir_y * attraction
            else:
                # Close to spot: settle down (reduce velocity, minimal wander)
                s.vx *= GRAZING_SETTLE_DAMP
                s.vy *= GRAZING_SETTLE_DAMP
                # Small correction to stay near spot
                correction = 0.02
                s.vx += (spot_x - s.x) * correction
                s.vy += (spot_y - s.y) * correction
            
            s.grazing_time += 1
            # Minimal wander only when settled (focused grazing)
            if dist_to_spot < GRAZING_SETTLE_RADIUS:
                s.vx += random.uniform(-GRAZING_WANDER, GRAZING_WANDER)
                s.vy += random.uniform(-GRAZING_WANDER, GRAZING_WANDER)
        else:
            # Outside grazing radius: move toward spot (directional)
            dir_x = (spot_x - s.x) / (dist_to_spot + 1e-6)
            dir_y = (spot_y - s.y) / (dist_to_spot + 1e-6)
            s.vx += dir_x * GRAZING_ATTRACTION * 0.5
            s.vy += dir_y * GRAZING_ATTRACTION * 0.5
            s.grazing_time = max(0, s.grazing_time - 1)
        
        # 5. DOG REPULSION: Avoid dealer pressure (herding toward pen)
        dx = s.x - dog_pos[0]
        dy = s.y - dog_pos[1]
        dog_dist = math.hypot(dx, dy) + 1e-6
        if dog_dist < DOG_HERDING_RADIUS:
            # Stronger repulsion when dog is close
            repulsion = (1.0 - dog_dist / DOG_HERDING_RADIUS) * DOG_HERDING_PUSH
            s.vx += (dx / dog_dist) * repulsion
            s.vy += (dy / dog_dist) * repulsion
            
            # When dog is pushing, also add slight push toward pen (greener pastures)
            # This makes sheep move away from spot when dog approaches
            if dist_to_spot < GRAZING_RADIUS:
                pen_dx = PEN.centerx - s.x
                pen_dy = PEN.centery - s.y
                pen_dist = math.hypot(pen_dx, pen_dy) + 1e-6
                # Small push toward pen when dog is herding
                push_toward_pen = SHEEP_PEN_PUSH * (1.0 - dog_dist / DOG_HERDING_RADIUS)
                s.vx += (pen_dx / pen_dist) * push_toward_pen
                s.vy += (pen_dy / pen_dist) * push_toward_pen
        
        # 6. BOUNDARY REPULSION: Prevent getting stuck in corners
        # Left boundary
        if s.x < BOUNDARY_MARGIN:
            s.vx += (BOUNDARY_MARGIN - s.x) / BOUNDARY_MARGIN * BOUNDARY_FORCE
        # Right boundary
        if s.x > W - BOUNDARY_MARGIN:
            s.vx -= (s.x - (W - BOUNDARY_MARGIN)) / BOUNDARY_MARGIN * BOUNDARY_FORCE
        # Top boundary
        if s.y < BOUNDARY_MARGIN:
            s.vy += (BOUNDARY_MARGIN - s.y) / BOUNDARY_MARGIN * BOUNDARY_FORCE
        # Bottom boundary
        if s.y > H - BOUNDARY_MARGIN:
            s.vy -= (s.y - (H - BOUNDARY_MARGIN)) / BOUNDARY_MARGIN * BOUNDARY_FORCE
        
        # 7. Speed limit and movement
        speed = math.hypot(s.vx, s.vy)
        max_speed = SHEEP_SPD * (1.5 if dist_to_spot < GRAZING_RADIUS else 1.0)
        if speed > max_speed:
            s.vx = (s.vx / speed) * max_speed
            s.vy = (s.vy / speed) * max_speed
        
        # Update position
        s.x += s.vx
        s.y += s.vy
        
        # Hard boundaries (safety net)
        s.x = max(5, min(W - 5, s.x))
        s.y = max(5, min(H - 5, s.y))

    def draw(s, screen, spot_x, spot_y):
        # Color based on grazing time (more yellow when grazing longer)
        dist_to_spot = math.hypot(s.x - spot_x, s.y - spot_y)
        if dist_to_spot < GRAZING_RADIUS:
            # Grazing - yellow tint based on time spent
            intensity = min(255, 180 + s.grazing_time * 2)
            c = (intensity, intensity, 150)
        elif in_pen(s.x, s.y):
            c = COL_SHEEP_IN_PEN
        else:
            c = COL_SHEEP
        pg.draw.circle(screen, c, (int(s.x), int(s.y)), 4)




class Dog:
    def __init__(d):
        d.x, d.y = W*0.15, H*0.15
        d.vx, d.vy = 0.0, 0.0
        d.tx, d.ty = d.x, d.y   # smoothed target

    def step(d, target, iv, breach):
        # --- smooth the target (reduces twitch when target jumps)
        d.tx += DOG_TARGET_SMOOTH * (target[0] - d.tx)
        d.ty += DOG_TARGET_SMOOTH * (target[1] - d.ty)

        # --- PD controller toward smoothed target
        ex, ey = d.tx - d.x, d.ty - d.y
        k_p = DOG_SPD_BASE * (1.0 + iv*0.02) * 0.60
        ax = k_p * ex - DOG_DAMP * d.vx
        ay = k_p * ey - DOG_DAMP * d.vy

        # --- accel clamp
        amag = (ax*ax + ay*ay) ** 0.5 + 1e-6
        if amag > DOG_A_MAX:
            ax *= DOG_A_MAX / amag
            ay *= DOG_A_MAX / amag

        # --- integrate velocity/position
        d.vx += ax; d.vy += ay
        # speed cap scales with IV
        vmax = DOG_SPD_BASE * (1.0 + iv*0.02) * 3.0
        vmag = (d.vx*d.vx + d.vy*d.vy) ** 0.5 + 1e-6
        if vmag > vmax:
            d.vx *= vmax / vmag
            d.vy *= vmax / vmag

        # --- near-target deadzone (bleed speed to stop oscillation)
        if (ex*ex + ey*ey) ** 0.5 < 12:
            d.vx *= 0.86; d.vy *= 0.86

        # --- move
        d.x += d.vx; d.y += d.vy

        # --- keep out of pen (radial shove + kill inward velocity)
        if PEN.collidepoint(d.x, d.y):
            cx, cy = PEN.center
            rx, ry = d.x - cx, d.y - cy
            r = (rx*rx + ry*ry) ** 0.5 + 1e-6
            # outward shove
            d.x += (rx / r) * 8.0
            d.y += (ry / r) * 8.0
            # zero any inward component
            inward = (d.vx*rx + d.vy*ry) / r
            if inward < 0:
                d.vx -= (rx / r) * inward
                d.vy -= (ry / r) * inward

        # --- bounds
        d.x = max(0, min(W, d.x)); d.y = max(0, min(H, d.y))

    def draw(d, screen):
        pg.draw.circle(screen, COL_DOG, (int(d.x), int(d.y)), 9)  # filled dot



def centroid(flock):
    x=sum(s.x for s in flock)/len(flock); y=sum(s.y for s in flock)/len(flock)
    return (x,y)

def farthest_from_pen(flock):
    cx,cy=PEN.center; best=None; bestd=-1
    for s in flock:
        d=math.hypot(s.x-cx,s.y-cy)
        if d>bestd: bestd=d; best=(s.x,s.y)
    return best

# ADD: farthest sheep OUTSIDE the pen (if any)
def farthest_outside_pen(flock):
    outside = [(s.x, s.y) for s in flock if not PEN.collidepoint(s.x, s.y)]
    if not outside: return None
    cx, cy = PEN.center
    return max(outside, key=lambda p: math.hypot(p[0]-cx, p[1]-cy))

def find_sheep_near_spot(flock, spot_x, spot_y, radius=GRAZING_RADIUS):
    """Find sheep that are grazing near the spot price"""
    return [s for s in flock if math.hypot(s.x - spot_x, s.y - spot_y) < radius]

def calculate_dog_herding_target(flock, spot_x, spot_y, pen_center):
    """
    Calculate where the dog should position to push grazing sheep toward the pen.
    The dog positions between the grazing sheep and the pen to create a herding effect.
    """
    grazing_sheep = find_sheep_near_spot(flock, spot_x, spot_y)
    
    if not grazing_sheep:
        # No sheep grazing, patrol near spot
        return (spot_x, spot_y)
    
    # Find the centroid of grazing sheep
    if len(grazing_sheep) > 0:
        avg_x = sum(s.x for s in grazing_sheep) / len(grazing_sheep)
        avg_y = sum(s.y for s in grazing_sheep) / len(grazing_sheep)
    else:
        avg_x, avg_y = spot_x, spot_y
    
    # Calculate direction from grazing sheep toward pen center
    dx = pen_center[0] - avg_x
    dy = pen_center[1] - avg_y
    dist = math.hypot(dx, dy) + 1e-6
    
    # Position dog between grazing sheep and pen (closer to sheep to push them)
    # This creates a herding effect: dog pushes sheep toward pen
    push_distance = min(80, dist * 0.4)  # Position 40% of the way toward pen
    target_x = avg_x + (dx / dist) * push_distance
    target_y = avg_y + (dy / dist) * push_distance
    
    return (target_x, target_y)


def draw_grassy_patch(screen, spot_x, spot_y, zones, sheep_count):
    """Draw the overgrazed grassy patch at spot price"""
    # Patch size depends on how many sheep are grazing
    patch_radius = 40 + min(30, sheep_count * 2)
    
    # Draw multiple layers for grass effect
    for i in range(3):
        radius = patch_radius - (i * 8)
        if radius > 0:
            # Color gets more "overgrazed" (brown) with more sheep
            green = max(50, 120 - sheep_count * 3)
            brown = min(100, 40 + sheep_count * 2)
            color = (brown, green, brown // 2)
            pg.draw.circle(screen, color, (int(spot_x), int(spot_y)), int(radius))
    
    # Draw grass texture (small lines)
    for _ in range(8):
        angle = random.uniform(0, math.pi * 2)
        dist = random.uniform(10, patch_radius - 5)
        x1 = spot_x + math.cos(angle) * dist
        y1 = spot_y + math.sin(angle) * dist
        x2 = x1 + math.cos(angle) * 3
        y2 = y1 + math.sin(angle) * 3
        pg.draw.line(screen, (80, 100, 50), (int(x1), int(y1)), (int(x2), int(y2)), 1)

def draw_altimeter(screen, zones, spot_x):
    """Draw horizontal altimeter showing spot price"""
    altimeter_y = H - 50  # Moved up to avoid controls hint
    altimeter_height = 20
    altimeter_width = W - 40
    altimeter_x = 20
    
    # Background
    pg.draw.rect(screen, (30, 30, 35), 
                 (altimeter_x, altimeter_y, altimeter_width, altimeter_height))
    pg.draw.rect(screen, (60, 60, 70), 
                 (altimeter_x, altimeter_y, altimeter_width, altimeter_height), 2)
    
    # Price range markers
    font_tiny = pg.font.SysFont(None, 12)
    price_min = zones['neg_gamma_range'][0]
    price_max = zones['pos_gamma_range'][1]
    price_range = price_max - price_min
    
    # Draw tick marks and labels
    num_ticks = 8
    for i in range(num_ticks + 1):
        tick_x = altimeter_x + (i / num_ticks) * altimeter_width
        price = price_min + (i / num_ticks) * price_range
        pg.draw.line(screen, (100, 100, 110), 
                    (int(tick_x), altimeter_y), 
                    (int(tick_x), altimeter_y + altimeter_height), 1)
        if i % 2 == 0:  # Label every other tick
            label = font_tiny.render(f"${price:.0f}", True, (180, 180, 190))
            screen.blit(label, (int(tick_x) - label.get_width()//2, altimeter_y - 14))
    
    # Spot price indicator (vertical line)
    if 0 <= spot_x <= W:
        alt_spot_x = altimeter_x + ((zones['spot'] - price_min) / price_range) * altimeter_width
        alt_spot_x = max(altimeter_x, min(altimeter_x + altimeter_width, alt_spot_x))
        pg.draw.line(screen, (255, 255, 100), 
                    (int(alt_spot_x), altimeter_y - 2), 
                    (int(alt_spot_x), altimeter_y + altimeter_height + 2), 3)
        
        # Spot price label
        font_small = pg.font.SysFont(None, 14)
        label = font_small.render(f"${zones['spot']:.2f}", True, (255, 255, 150))
        screen.blit(label, (int(alt_spot_x) - label.get_width()//2, altimeter_y - 28))
    
    # Gamma flip marker
    flip_x = altimeter_x + ((zones['gamma_flip'] - price_min) / price_range) * altimeter_width
    if altimeter_x <= flip_x <= altimeter_x + altimeter_width:
        pg.draw.line(screen, (200, 200, 100), 
                    (int(flip_x), altimeter_y), 
                    (int(flip_x), altimeter_y + altimeter_height), 2)

def draw_zones(screen, breach, zones):
    """Draw gamma regime zones with price labels"""
    if not zones:
        return
    
    font_small = pg.font.SysFont(None, 16)
    font_med = pg.font.SysFont(None, 18)
    font_large = pg.font.SysFont(None, 22)
    
    # Draw negative gamma zone (below flip) - LEFT side
    neg_x = price_to_screen_x(zones['neg_gamma_range'][0], zones, W)
    flip_x = price_to_screen_x(zones['gamma_flip'], zones, W)
    if neg_x < flip_x:
        pg.draw.rect(screen, COL_NEG_GAMMA, pg.Rect(0, 0, flip_x, H))
        # Label
        label = font_med.render("NEGATIVE GAMMA", True, (255, 200, 200))
        screen.blit(label, (10, H - 40))
        label2 = font_small.render("(Dealers Short Γ)", True, (200, 150, 150))
        screen.blit(label2, (10, H - 25))
        label3 = font_small.render("Price Accelerates", True, (200, 150, 150))
        screen.blit(label3, (10, H - 10))
    
    # Draw positive gamma zone (above flip) - RIGHT side
    pos_x = price_to_screen_x(zones['pos_gamma_range'][1], zones, W)
    if flip_x < pos_x:
        pg.draw.rect(screen, COL_POS_GAMMA, pg.Rect(flip_x, 0, pos_x - flip_x, H))
        # Label
        label = font_med.render("POSITIVE GAMMA", True, (200, 255, 200))
        screen.blit(label, (W - 180, H - 40))
        label2 = font_small.render("(Dealers Long Γ)", True, (150, 200, 150))
        screen.blit(label2, (W - 180, H - 25))
        label3 = font_small.render("Price Pinned", True, (150, 200, 150))
        screen.blit(label3, (W - 180, H - 10))
    
    # Draw breach overlays
    if breach == BREACH_PUT:
        pg.draw.rect(screen, COL_PUT, pg.Rect(0, 0, W//2, H), 3)
    elif breach == BREACH_CALL:
        pg.draw.rect(screen, COL_CALL, pg.Rect(W//2, 0, W//2, H), 3)
    
    # Draw gamma-neutral pen (center zone)
    pg.draw.rect(screen, COL_PEN_BORDER, PEN, 3, border_radius=14)
    pg.draw.rect(screen, COL_PEN, PEN, border_radius=14)
    
    # Pen label
    label = font_med.render("GAMMA-NEUTRAL ZONE", True, (200, 200, 210))
    screen.blit(label, (PEN.centerx - label.get_width()//2, PEN.top - 25))
    label2 = font_small.render("(Price Pinned Here)", True, (180, 180, 200))
    screen.blit(label2, (PEN.centerx - label2.get_width()//2, PEN.top - 10))
    
    # Price level markers
    flip_x = price_to_screen_x(zones['gamma_flip'], zones, W)
    spot_x = price_to_screen_x(zones['spot'], zones, W)
    put_x = price_to_screen_x(zones['put_wall'], zones, W)
    call_x = price_to_screen_x(zones['call_wall'], zones, W)
    
    # Gamma flip line
    if 0 <= flip_x <= W:
        pg.draw.line(screen, (255, 255, 100), (flip_x, 0), (flip_x, H), 2)
        label = font_small.render(f"Flip: ${zones['gamma_flip']:.2f}", True, (255, 255, 150))
        screen.blit(label, (flip_x - label.get_width()//2, 5))
    
    # Spot price line
    if 0 <= spot_x <= W:
        pg.draw.line(screen, (255, 255, 255), (spot_x, 0), (spot_x, H), 1)
        label = font_small.render(f"Spot: ${zones['spot']:.2f}", True, (255, 255, 255))
        screen.blit(label, (spot_x - label.get_width()//2, 25))
    
    # Wall markers
    if 0 <= put_x <= W:
        pg.draw.line(screen, (255, 100, 100), (put_x, 0), (put_x, H), 1)
        label = font_small.render(f"Put Wall: ${zones['put_wall']:.2f}", True, (255, 150, 150))
        screen.blit(label, (put_x - label.get_width()//2, H - 60))
    
    if 0 <= call_x <= W:
        pg.draw.line(screen, (100, 255, 100), (call_x, 0), (call_x, H), 1)
        label = font_small.render(f"Call Wall: ${zones['call_wall']:.2f}", True, (150, 255, 150))
        screen.blit(label, (call_x - label.get_width()//2, H - 60))


def main(ticker_override=None, data_dir_override=None):
    pg.init()
    screen = pg.display.set_mode((W, H))
    pg.display.set_caption("Dealer Shepherd — Gamma Neutral Herding")
    clock = pg.time.Clock()

    # Initialize ticker: --ticker overrides config default
    ticker = (ticker_override or TICKER_DEFAULT).strip().upper()
    base_dir = (data_dir_override or "").strip() or None
    historical_df = None
    backtest_mode = False
    backtest_index = 0
    backtest_auto_play_counter = 0

    def _load_stock(t):
        return load_stock_data(t, base_dir=base_dir) if base_dir else load_stock_data(t)

    def _latest_price(t):
        return get_latest_price(t, base_dir=base_dir) if base_dir else get_latest_price(t)

    # Load historical data for backtesting
    if BACKTEST_ENABLED:
        try:
            historical_df = _load_stock(ticker)
            print(f"Loaded {len(historical_df)} historical records for {ticker}")
        except Exception as e:
            print(f"Warning: Could not load historical data: {e}")
            historical_df = None

    try:
        spot_price = _latest_price(ticker)
        print(f"Loaded {ticker} price: ${spot_price:.2f}")
    except Exception as e:
        print(f"Error loading price for {ticker}: {e}")
        print("Tip: Use --ticker SPY and --data-dir <path> (e.g. folder with <TICKER>.csv)")
        print("Using fallback price: $500.00")
        spot_price = 500.0

    zones = calculate_gamma_zones(spot_price)
    base_spot_price = spot_price  # Store original price
    base_zones = zones.copy()  # Store base zones for recovery
    print(f"Gamma zones: Flip=${zones['gamma_flip']:.2f}, "
          f"Put Wall=${zones['put_wall']:.2f}, Call Wall=${zones['call_wall']:.2f}")
    
    rng = lambda: [Sheep() for _ in range(N)]
    flock = rng()
    dog = Dog()
    breach = BREACH_NONE
    iv = IV_DEFAULT
    frame_count = 0
    regime_state = RegimeState.RANGE_BOUND

    # Breach confirmation system
    breach_pending = False  # True when waiting for L/S confirmation
    breach_duration = 0  # How long breach has been active

    if ENABLE_LOGGING:
        init_logger()

    while True:
        for e in pg.event.get():
            if e.type == pg.QUIT:
                pg.quit()
                sys.exit()
            if e.type == pg.KEYDOWN:
                if e.key == pg.K_ESCAPE:
                    if breach_pending:
                        breach_pending = False  # Cancel breach confirmation
                    else:
                        pg.quit()
                        sys.exit()
                if e.key == pg.K_SPACE:
                    if not breach_pending and breach == BREACH_NONE:
                        breach_pending = True  # Initiate breach confirmation
                if e.key == pg.K_l and breach_pending:
                    # Long breach: price goes above flip point
                    breach = BREACH_CALL
                    breach_pending = False
                    breach_duration = 0
                    # Store current flip before moving price
                    current_flip = zones['gamma_flip']
                    strike_spacing = zones['strike_spacing']
                    # Move price above flip
                    spot_price = current_flip + strike_spacing * BREACH_MAGNITUDE
                    zones = calculate_gamma_zones(spot_price)
                    print(f"LONG BREACH: Price moved to ${spot_price:.2f} (above flip ${current_flip:.2f})")
                if e.key == pg.K_s and breach_pending:
                    # Short breach: price collapses below flip point
                    breach = BREACH_PUT
                    breach_pending = False
                    breach_duration = 0
                    # Store current flip before moving price
                    current_flip = zones['gamma_flip']
                    strike_spacing = zones['strike_spacing']
                    # Move price below flip
                    spot_price = current_flip - strike_spacing * BREACH_MAGNITUDE
                    zones = calculate_gamma_zones(spot_price)
                    print(f"SHORT BREACH: Price collapsed to ${spot_price:.2f} (below flip ${current_flip:.2f})")
                if e.key in (pg.K_r,):
                    flock = rng()
                    dog = Dog()
                    iv = IV_DEFAULT
                    breach = BREACH_NONE
                    breach_pending = False
                    breach_duration = 0
                    frame_count = 0
                    regime_state = RegimeState.RANGE_BOUND
                    # Reset to base price
                    spot_price = base_spot_price
                    zones = calculate_gamma_zones(spot_price)
                    base_zones = zones.copy()  # Update base zones
                if e.key in (pg.K_EQUALS, pg.K_PLUS):
                    iv = min(IV_MAX, iv + IV_STEP)
                if e.key == pg.K_MINUS:
                    iv = max(IV_MIN, iv - IV_STEP)
                if e.key == pg.K_t:
                    # Toggle ticker
                    idx = (TICKERS.index(ticker) + 1) % len(TICKERS) if ticker in TICKERS else 0
                    ticker = TICKERS[idx]
                    try:
                        spot_price = _latest_price(ticker)
                        base_spot_price = spot_price
                        zones = calculate_gamma_zones(spot_price)
                        base_zones = zones.copy()
                        breach = BREACH_NONE
                        breach_pending = False
                        breach_duration = 0
                        # Reload historical data for new ticker
                        if BACKTEST_ENABLED:
                            try:
                                historical_df = _load_stock(ticker)
                                if backtest_mode:
                                    backtest_index = len(historical_df) - 1
                                print(f"Loaded {len(historical_df)} historical records for {ticker}")
                            except Exception as ex:
                                print(f"Warning: Could not load historical data: {ex}")
                                historical_df = None
                        print(f"Switched to {ticker}: ${spot_price:.2f}")
                    except Exception as ex:
                        print(f"Error loading {ticker}: {ex}")
                if e.key == pg.K_p:
                    # Print current zones
                    print(f"\n=== {ticker} Gamma Zones ===")
                    print(f"Spot Price: ${zones['spot']:.2f}")
                    print(f"Gamma Flip: ${zones['gamma_flip']:.2f}")
                    print(f"Put Wall: ${zones['put_wall']:.2f}")
                    print(f"Call Wall: ${zones['call_wall']:.2f}")
                    print(f"Negative Gamma Range: ${zones['neg_gamma_range'][0]:.2f} - ${zones['neg_gamma_range'][1]:.2f}")
                    print(f"Positive Gamma Range: ${zones['pos_gamma_range'][0]:.2f} - ${zones['pos_gamma_range'][1]:.2f}")
                    print("=" * 30)
                if e.key == pg.K_b and BACKTEST_ENABLED and historical_df is not None:
                    # Toggle backtest mode
                    backtest_mode = not backtest_mode
                    if backtest_mode:
                        # Start at most recent data
                        backtest_index = len(historical_df) - 1
                        price, date = get_historical_price(historical_df, backtest_index)
                        if price:
                            spot_price = price
                            zones = calculate_gamma_zones(spot_price)
                            print(f"Backtest mode ON: {date} - ${spot_price:.2f}")
                    else:
                        # Return to latest price
                        try:
                            spot_price = _latest_price(ticker)
                            zones = calculate_gamma_zones(spot_price)
                            print("Backtest mode OFF: Using latest price")
                        except:
                            pass
                if e.key == pg.K_LEFT and backtest_mode and historical_df is not None:
                    # Move backward in history
                    backtest_index = max(0, backtest_index - BACKTEST_STEP_DAYS)
                    price, date = get_historical_price(historical_df, backtest_index)
                    if price:
                        spot_price = price
                        zones = calculate_gamma_zones(spot_price)
                if e.key == pg.K_RIGHT and backtest_mode and historical_df is not None:
                    # Move forward in history
                    backtest_index = min(len(historical_df) - 1, backtest_index + BACKTEST_STEP_DAYS)
                    price, date = get_historical_price(historical_df, backtest_index)
                    if price:
                        spot_price = price
                        zones = calculate_gamma_zones(spot_price)
                if e.key == pg.K_HOME and backtest_mode and historical_df is not None:
                    # Jump to earliest data
                    backtest_index = 0
                    price, date = get_historical_price(historical_df, backtest_index)
                    if price:
                        spot_price = price
                        zones = calculate_gamma_zones(spot_price)
                if e.key == pg.K_END and backtest_mode and historical_df is not None:
                    # Jump to latest data
                    backtest_index = len(historical_df) - 1
                    price, date = get_historical_price(historical_df, backtest_index)
                    if price:
                        spot_price = price
                        zones = calculate_gamma_zones(spot_price)

        # Auto-play through history if enabled
        if backtest_mode and BACKTEST_AUTO_PLAY and historical_df is not None:
            backtest_auto_play_counter += 1
            if backtest_auto_play_counter >= BACKTEST_AUTO_PLAY_SPEED:
                backtest_auto_play_counter = 0
                # Move forward, loop back to start if at end
                backtest_index = (backtest_index + BACKTEST_STEP_DAYS) % len(historical_df)
                price, date = get_historical_price(historical_df, backtest_index)
                if price:
                    spot_price = price
                    zones = calculate_gamma_zones(spot_price)

        screen.fill(COL_BG)
        draw_zones(screen, breach, zones)

        # Calculate spot price position on screen
        spot_x = price_to_screen_x(zones['spot'], zones, W)
        spot_y = H // 2  # Spot price is at middle of screen vertically
        
        # Count sheep in grazing area (before updating, use previous frame count for drawing)
        grazing_sheep = [s for s in flock if math.hypot(s.x - spot_x, s.y - spot_y) < GRAZING_RADIUS]
        sheep_in_grazing = len(grazing_sheep)
        
        # Draw grassy patch (grazing area) at spot price - draw before sheep so they appear on top
        draw_grassy_patch(screen, spot_x, spot_y, zones, sheep_in_grazing)
        
        cen = centroid(flock)
        
        # Breach auto-recovery: after max duration, price returns to neutral
        if breach != BREACH_NONE:
            breach_duration += 1
            if breach_duration >= BREACH_MAX_DURATION:
                # Gradually return price toward base flip (not current zones flip)
                target_flip = base_zones['gamma_flip']
                if breach == BREACH_CALL:
                    # Long breach: bring price down toward base flip
                    spot_price = spot_price * (1 - BREACH_RECOVERY_SPEED) + target_flip * BREACH_RECOVERY_SPEED
                else:  # BREACH_PUT
                    # Short breach: bring price up toward base flip
                    spot_price = spot_price * (1 - BREACH_RECOVERY_SPEED) + target_flip * BREACH_RECOVERY_SPEED
                
                # Check if close enough to base flip to end breach
                if abs(spot_price - target_flip) < base_zones['strike_spacing'] * 0.3:
                    breach = BREACH_NONE
                    breach_duration = 0
                    spot_price = base_spot_price
                    zones = base_zones.copy()
                    print("Breach resolved: Price returned to base")
                else:
                    zones = calculate_gamma_zones(spot_price)

        # Regime scaffold: map breach to RANGE_BOUND / BREACH_UP / BREACH_DOWN
        if breach == BREACH_NONE:
            regime_state = RegimeState.RANGE_BOUND
        elif breach == BREACH_CALL:
            regime_state = RegimeState.BREACH_UP
        else:
            regime_state = RegimeState.BREACH_DOWN

        # Dog target: actively herd sheep away from spot price toward pen
        pen_center = (PEN.centerx, PEN.centery)
        
        if breach != BREACH_NONE:
            # During breach, dog applies gentle pressure to guide price back
            # Target area opposite the breach direction
            if breach == BREACH_CALL:
                # Long breach: guide price down (target below spot)
                t = (spot_x, spot_y + 80)
            else:  # BREACH_PUT
                # Short breach: guide price up (target above spot)
                t = (spot_x, spot_y - 80)
        else:
            # Normal operation: dog herds grazing sheep toward pen (greener pastures)
            t = calculate_dog_herding_target(flock, spot_x, spot_y, pen_center)
        
        target = t

        # Update dog first (dealer repositions), then sheep react
        dog.step(target, iv, breach)
        
        # Update sheep with flocking and grazing behavior
        for s in flock:
            s.step(flock, spot_x, spot_y, (dog.x, dog.y), zones)

        if ENABLE_LOGGING and frame_count % LOG_EVERY_N_FRAMES == 0:
            n_grazing = sum(1 for s in flock if math.hypot(s.x - spot_x, s.y - spot_y) < GRAZING_RADIUS)
            n_in_pen = sum(1 for s in flock if in_pen(s.x, s.y))
            log_frame(frame_count, ticker, zones, iv, breach, flock, dog, n_grazing, n_in_pen, regime_state)

        # Draw
        for s in flock:
            s.draw(screen, spot_x, spot_y)
        dog.draw(screen)
        
        # Draw horizontal altimeter
        draw_altimeter(screen, zones, spot_x)
        
        # Draw backtest slider if in backtest mode
        if backtest_mode and historical_df is not None:
            current_date = None
            price, date = get_historical_price(historical_df, backtest_index)
            if date is not None:
                # Format date for display
                if hasattr(date, 'strftime'):
                    current_date = date.strftime('%Y-%m-%d')
                else:
                    current_date = str(date)
            draw_backtest_slider(screen, backtest_index, len(historical_df), current_date)
        
        # Enhanced HUD
        font = pg.font.SysFont(None, 20)
        font_small = pg.font.SysFont(None, 16)
        
        # Count sheep in different areas
        in_pen_count = sum(1 for s in flock if in_pen(s.x, s.y))
        
        # Adjust HUD position if backtest slider is visible
        hud_y_offset = 50 if (backtest_mode and historical_df is not None) else 10
        
        regime_label = regime_state.name.replace("_", "-")  # RANGE_BOUND -> RANGE-BOUND
        hud_lines = [
            f"Ticker: {ticker} | Spot: ${zones['spot']:.2f} | IV: {iv}%",
            f"Breach: {('None', 'SHORT', 'LONG')[breach]} | Total: {len(flock)} | Grazing: {sheep_in_grazing} | In Pen: {in_pen_count}",
            f"Gamma Flip: ${zones['gamma_flip']:.2f} | Range: ${zones['neg_gamma_range'][0]:.2f} - ${zones['pos_gamma_range'][1]:.2f}",
            f"Regime: {regime_label}"
        ]

        if backtest_mode:
            hud_lines.append(f"BACKTEST MODE {'[AUTO-PLAY]' if BACKTEST_AUTO_PLAY else ''}")
        
        for i, line in enumerate(hud_lines):
            screen.blit(font.render(line, True, COL_TEXT), (12, hud_y_offset + i * 22))
        
        # Breach confirmation prompt
        if breach_pending:
            prompt_font = pg.font.SysFont(None, 24)
            prompt = prompt_font.render("BREACH CONFIRMATION: Press L (Long) or S (Short), ESC to cancel", 
                                       True, (255, 200, 100))
            prompt_rect = prompt.get_rect(center=(W//2, H//2))
            # Draw background for prompt
            pg.draw.rect(screen, (40, 40, 45), 
                        (prompt_rect.x - 10, prompt_rect.y - 5, 
                         prompt_rect.width + 20, prompt_rect.height + 10), 
                        border_radius=8)
            screen.blit(prompt, prompt_rect)
        
        # Controls hint (centered at bottom)
        controls = "SPACE=breach | R=reset | +/-=IV | T=ticker | P=print zones"
        if BACKTEST_ENABLED and historical_df is not None:
            controls += " | B=backtest | ←→=navigate | HOME/END=jump"
        controls += " | ESC=quit"
        controls_text = font_small.render(controls, True, (150, 150, 150))
        controls_x = (W - controls_text.get_width()) // 2
        screen.blit(controls_text, (controls_x, H - 18))

        frame_count += 1
        pg.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sheep herding (options) — dealer/retail sim.")
    ap.add_argument("--ticker", default=None, help="Ticker to load (e.g. SPY, QQQ). Overrides config default.")
    ap.add_argument("--data-dir", default=None, help="Override data_loader base dir (folder containing <TICKER>.csv).")
    args = ap.parse_args()
    main(ticker_override=args.ticker, data_dir_override=args.data_dir)
