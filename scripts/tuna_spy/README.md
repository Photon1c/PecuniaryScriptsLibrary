# SPY OI Gravity Well Simulator

A Pygame thought-experiment visualizing how **open interest** at options strikes
creates gravitational fields that attract and repel price.

## Concept

Each options strike with significant open interest acts as a **gravity well**:

- **Call OI** above current price → ceiling / resistance (price gets capped)
- **Put OI** below current price → floor / support (price gets supported)
- **Balanced strikes** (equal call/put OI) → strong magnetic attractors
- **Max Pain strike** → the strongest attractor (where most OI expires worthless)
- **ATM gamma-rich strikes** → most volatile, highest energy zone

Price particles undergo Brownian motion + net OI-field acceleration. The
ensemble distribution shows you the probability well the market is sitting in.

### Axes (how to read the chart)

**Vertical (Y)** is **price**: each probe’s implied price maps directly to height.

**Horizontal (X)** is **crowd structure in the liquidity field**, not time and not a second ticker. Probes that sit near the same **price level** (within a few dollars) form a *local crowd*: they **cohere** toward that crowd’s average horizontal position, **separate** when they overlap in screen space, and **compress** inward when many peers stack up—so the blob can tighten horizontally while price still moves on Y. In `--live` mode, **Volume** columns from your stock and options CSVs (mirrored call/put `Volume` when present) gently scale how tight that cohesion and compression run, as a proxy for how “busy” the tape is.

The framing: large participants do not need to move *along* these axes; the structure of the crowd—density and vulnerability in price space—is what the field makes visible.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
python main.py --live
```

## Controls

| Key | Action |
|-----|--------|
| `R` | Reset all particles |
| `N` | Inject noise shock (simulate news event) |
| `E` | Force gamma expiry event |
| `+` / `-` | Increase / decrease noise level |
| `Q` / `Esc` | Quit |

## What you see

| Visual element | Meaning |
|---|---|
| **Teal bars (right)** | Call open interest at that strike |
| **Red bars (left)** | Put open interest at that strike |
| **Gold glow** | Max Pain strike |
| **Purple glow** | ATM gamma-rich strike (nearest to current price) |
| **Field strip (far left)** | Net gravity direction: green=upward pull, red=downward pull |
| **Blue particles** | Price probes drifting through the field |
| **Hot orange particles** | High-velocity / high-momentum particles |
| **White spine** | Ensemble centroid — the "market path" |
| **Phase badge** | Current simulation phase |
| **Yellow spine** | Path of the ensemble centroid (mean X and mean Y of probes) |

## Simulation Phases

1. **DISPERSED** — particles spread, low density
2. **CLUSTERING** — particles converging on a gravity well
3. **PINNED** — majority cluster around dominant strike (gamma pinning)
4. **EXPIRY CRUSH** — expiry event fires, OI wells reset, chaos spike
5. **REBALANCE** — system finds new equilibrium

## Architecture

```
spy_gravity/
├── main.py              ← entry point + game loop
├── config.py            ← all constants, OI data, colors
├── entities/
│   └── particle.py      ← PriceParticle (Brownian + field-driven)
├── systems/
│   ├── liquidity_field.py  ← OI gravity math, field cache
│   ├── crowd_structure.py  ← horizontal “crowd density” integration (X axis)
│   ├── metrics.py          ← cluster detection, gamma exposure
│   └── phase_manager.py    ← phase state machine
├── visuals/
│   └── draw_all.py         ← all rendering
└── utils/
    └── vectors.py          ← math helpers
```

## Customizing OI data

Edit `STRIKE_OI` in `config.py`:
```python
STRIKE_OI = {
    strike: (call_oi_thousands, put_oi_thousands),
    ...
}
```

Change `SPY_CURRENT` to move where particles start.
Adjust `GRAVITY_FALLOFF` to make wells sharper (higher) or broader (lower).
