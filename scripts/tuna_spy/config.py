# config.py — SPY Gravity Well Simulator

# ── Screen ──────────────────────────────────────────────────────────────────
WIDTH  = 1280
HEIGHT = 800
FPS    = 60
TITLE  = "SPY OI Gravity Wells — Price Field Simulator"

# Left / top / bottom padding for price↔Y mapping (room for ruler + labels)
CHART_MARGIN = 68

# ── SPY Price Universe ───────────────────────────────────────────────────────
SPY_CURRENT   = 710.0   # hypothetical current price
SPY_RANGE_LOW  = 650.0
SPY_RANGE_HIGH = 770.0

# ── Open Interest Strike Map ─────────────────────────────────────────────────
# Format: { strike: (call_oi, put_oi) }
# Units: thousands of contracts
# These are plausible synthetic values for the thought experiment
STRIKE_OI = {
    660: (12,  85),   # heavy put wall
    665: (8,   60),
    670: (18,  95),   # big put wall
    675: (22,  40),
    680: (35,  55),
    685: (28,  38),
    690: (60,  70),   # balanced zone — gravitational equilibrium
    695: (75,  35),
    700: (150, 90),   # massive call/put crossover — MAX PAIN ZONE
    705: (90,  45),
    710: (110, 30),   # current price — gamma-rich strike
    715: (85,  22),
    720: (130, 18),   # major call wall
    725: (70,  15),
    730: (95,  12),
    735: (55,  8),
    740: (80,  10),   # secondary call wall
    745: (40,  6),
    750: (65,  8),
    755: (30,  4),
    760: (45,  5),
}

# ── Gravity Physics ───────────────────────────────────────────────────────────
GRAVITY_FALLOFF   = 2.2   # distance exponent (higher = sharper wells)
CALL_GRAVITY      = 0.55  # calls repel price above strike, attract below
PUT_GRAVITY       = 0.55  # puts repel price below strike, attract above
COMBINED_SCALE    = 0.012 # overall field strength scaler
MAX_ACCEL         = 0.8   # cap on acceleration per frame
DAMPING           = 0.94  # velocity damping (friction)
NOISE_STRENGTH    = 0.15  # random walk noise injected each frame

# ── Particle (price probe) config ────────────────────────────────────────────
NUM_PARTICLES      = 80
PARTICLE_MASS_MIN  = 0.6
PARTICLE_MASS_MAX  = 1.4
PARTICLE_ALPHA_BASE = 180

# ── Phase thresholds ─────────────────────────────────────────────────────────
CLUSTER_RADIUS     = 12.0   # px: particles within this are "clustered"
CLUSTER_THRESHOLD  = 0.55   # fraction of particles in dominant cluster
GAMMA_EXPIRY_TICKS = 3600   # ticks before "expiry" event fires

# ── Crowd structure (horizontal = density / positioning in price-local crowd) ─
# Y is always price. X emerges from cohesion + separation + compression among
# particles whose *prices* lie within CROWD_NEIGHBOR_PRICE_SPAN dollars.
CROWD_NEIGHBOR_PRICE_SPAN       = 4.5   # $ — who counts as your local crowd
CROWD_COHESION_BASE             = 0.048
CROWD_SEPARATION_BASE          = 1.18
CROWD_SEPARATION_RADIUS        = 46.0  # px — push apart when closer in X
CROWD_COMPRESSION_BASE         = 0.034  # inward pull when many local peers
CROWD_COMPRESSION_MIN_NEIGHBORS = 3
CROWD_HORIZONTAL_DAMP          = 0.90
CROWD_NOISE                    = 0.07
CROWD_X_MARGIN_LO_FRAC         = 0.08
CROWD_X_MARGIN_HI_FRAC         = 0.92

# Patched by --live from stock/options Volume columns when available (else 1.0)
LIVE_CROWD_COHESION_SCALE      = 1.0
LIVE_CROWD_COMPRESSION_SCALE   = 1.0
LIVE_LAST_STOCK_VOLUME         = 0.0
LIVE_OPTION_VOLUME_CALL_SUM    = 0.0
LIVE_OPTION_VOLUME_PUT_SUM     = 0.0

# ── Colors (R, G, B) ─────────────────────────────────────────────────────────
C_BG           = (6,   8,  18)
C_GRID         = (20,  28,  50)
C_PRICE_LINE   = (180, 200, 255)
C_CALL_WELL    = (40,  220, 160)   # teal-green
C_PUT_WELL     = (220,  60,  90)   # crimson
C_BALANCED     = (200, 180,  60)   # gold — balanced strike
C_PARTICLE     = (140, 200, 255)   # cool blue
C_PARTICLE_HOT = (255, 180,  60)   # heated / pinned particle
C_SPINE        = (255, 255, 200)   # price ribbon
C_TEXT         = (200, 215, 255)
C_ACCENT       = (100, 140, 255)
C_MAX_PAIN     = (255, 220,  60)   # gold halo on max pain strike
C_GAMMA_RICH   = (160, 100, 255)   # purple — current ATM

# ── Font sizes ───────────────────────────────────────────────────────────────
FONT_SMALL  = 13
FONT_MED    = 16
FONT_LARGE  = 22
FONT_TITLE  = 28
