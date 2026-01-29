# Sheep Herding Simulation — Structured Summary for ChatGPT

**Files:** `sheep_herding_stocks.py` | `sheep_herding_options.py`  
**Context:** Pygame “sheep herding” simulations used as an analogy for dealer–retail market microstructure.

---

## 1. HIGH-LEVEL PURPOSE

- **What each script simulates:** Both scripts run a real-time Pygame simulation where a **dog** herds **sheep** around a **field**. Sheep are attracted to a **grassy patch** (spot price) and optionally **pen** (gamma-neutral zone). The dog repels sheep near it and is steered to push grazing sheep toward the pen. The user can trigger **breaches** (Long / Short), which move “price” above or below a **gamma flip**, then auto-recover toward the base level. **Implied volatility (IV)** is adjustable and scales dog speed/agility. **Backtest mode** lets you step through historical OHLC (e.g. SPY) and map price to gamma zones.

- **Sheep-herding → market metaphor:**
  - **Sheep** = retail order flow / speculative interest (directionally biased, flocking).
  - **Dog** = dealer / market maker (contains flow, herds price toward gamma-neutral zone).
  - **Grassy patch (grazing area)** = spot price; retail “grazes” there (converges to current price).
  - **Pen** = gamma-neutral zone; dealer wants flow/price pinned there.
  - **Field** = price range; screen X maps to price via `neg_gamma_range`–`pos_gamma_range`.
  - **Fence / walls** = put wall, call wall, gamma flip (support/resistance implied by dealer gamma).
  - **Breach (L/S)** = forced move beyond flip (e.g. breakout or flush), then mean-reversion back.

- **Stocks vs options:** The two files are **structurally identical** in the repo (only the first-line comment differs). The *options* framing is explicit: gamma flip, put/call walls, IV, strike spacing. The *stocks* version was likely intended as an equity-flow–only analogue (same mechanic, simpler mental model: dealer contains retail around spot). Today, both implement the same gamma-aware, IV-scaled herding logic.

---

## 2. ENTITY & CONCEPT MAP

| Entity | Market meaning | Key attributes | Vol / IV / gamma / flow |
|--------|----------------|----------------|--------------------------|
| **Sheep** | Retail flow / speculative orders | `x,y` pos; `vx,vy` vel; `grazing_time`; speed cap `SHEEP_SPD`; attraction to spot, repulsion from dog | Grazing → clustering at spot (flow intensity). Speed bump in grazing radius ≈ “aggressiveness” near spot. |
| **Dog** | Dealer / MM | `x,y`; `vx,vy`; smoothed target `tx,ty`; PD control (`k_p`, `DOG_DAMP`); `DOG_HERDING_RADIUS`, `DOG_HERDING_PUSH` | **IV** scales `k_p` and speed cap: higher IV ⇒ faster, more reactive dealer. Cannot enter pen. |
| **Pen** | Gamma-neutral zone | `PEN` rect (center ≈ flip); fixed `PEN_WIDTH` × `PEN_HEIGHT` | Where dealer wants price; sheep can enter; dog herds toward it. |
| **Grassy patch** | Spot price (grazing) | `GRAZING_RADIUS`, `GRAZING_SETTLE_RADIUS`, `GRAZING_ATTRACTION`, `GRAZING_WANDER`, `GRAZING_SETTLE_DAMP` | Stronger attraction ⇒ retail clusters more at spot. Overgrazed visual = many sheep at spot (flow concentration). |
| **Gamma zones** | Regime (dealers long/short Γ) | `gamma_flip`, `put_wall`, `call_wall`, `neg_gamma_range`, `pos_gamma_range`; `STRIKE_SPACING` | **Gamma flip** ≈ ATM; **put/call walls** ≈ key strikes; **neg gamma** (left) = price accelerates down; **pos gamma** (right) = price pinned. |
| **Breach** | Forced breakout / flush | `BREACH_PUT` / `BREACH_CALL`; `BREACH_MAGNITUDE` (strikes), `BREACH_MAX_DURATION`, `BREACH_RECOVERY_SPEED` | **Rupture** past flip; then **mean reversion** toward base flip. L = call breach; S = put breach. |
| **Spot** | Current price | `zones['spot']`; mapped to `spot_x`, `spot_y` (screen) | Drives grazing center and altimeter; backtest overrides from history. |
| **Flock** | Aggregate retail | Centroid, “farthest from pen,” “sheep near spot” | Used for dog target (herd grazing sheep toward pen) and HUD (grazing count, in-pen count). |

- **Flocking (separation, alignment, cohesion):** Neighbor-based rules with `FLOCK_RADIUS`, `FLOCK_SEPARATION`, `FLOCK_ALIGNMENT`, `FLOCK_COHESION`. Model “retail herds” — correlated flow, not independent particles.

---

## 3. SIMULATION LOOP & LOGIC

- **Per-frame updates (60 fps):**
  1. **Input:** Key events (breach L/S, reset, IV +/-, ticker, backtest step, etc.).
  2. **Breach recovery:** If breach active, tick `breach_duration`; once ≥ `BREACH_MAX_DURATION`, interpolate spot toward `base_zones['gamma_flip']` at `BREACH_RECOVERY_SPEED` until near flip, then clear breach and restore base.
  3. **Dog target:**  
     - **No breach:** `calculate_dog_herding_target` → centroid of sheep near spot, then target 40% toward pen from that centroid (dog between grazing cluster and pen).  
     - **Breach:** target offset from spot opposite breach (call breach ⇒ below spot; put breach ⇒ above spot) to “guide” price back.
  4. **Dog step:** EMA-smooth target; PD control toward it; accel clamp; velocity cap (IV-scaled); keep out of pen (radial shove + kill inward velocity); clamp to screen.
  5. **Sheep step:** For each sheep: separation/alignment/cohesion ⇒ grazing (attract to spot, settle near it, wander) ⇒ dog repulsion (+ optional “greener pastures” push toward pen when dog close) ⇒ boundary repulsion ⇒ speed limit ⇒ integrate position.
  6. **Draw:** Zones, pen, patch, sheep, dog, altimeter, backtest slider, HUD, breach prompt, controls hint.

- **Dog–sheep interaction:**
  - **Push:** Dog exerts repulsion within `DOG_HERDING_RADIUS`; strength ∝ `(1 - dist/radius) * DOG_HERDING_PUSH`. When sheep is grazing *and* dog close, add `SHEEP_PEN_PUSH` toward pen (“greener pastures”) so they’re driven away from spot toward pen.
  - **Containment:** Dog targets between grazing cluster and pen, so it naturally positions to push sheep from spot toward gamma-neutral zone. No explicit “panic” state; reaction is purely repulsion + pen push.

- **Market-correspondence:**
  - **MM range-keeping:** Dog herding grazing sheep toward pen ≈ dealer containing retail around gamma-neutral zone.
  - **Lure-and-gate:** Spot attracts sheep; dog then pushes them toward pen. Can be read as retail drawn to spot, then dealer pushing flow back into the “safe” zone.
  - **Ruptures:** Breach = exogenous move past flip; recovery = mean reversion. Dog “guides” during breach by targeting opposite side of spot.

---

## 4. CONTROLS, PARAMETERS, AND OUTPUTS

- **Important user-adjustable parameters:**
  - **IV:** `+` / `-` → `IV_MIN`..`IV_MAX` step `IV_STEP`. Scales dog gain and speed.
  - **Breach:** `Space` → confirm → `L` (call) / `S` (put). Magnitude and recovery: `BREACH_MAGNITUDE`, `BREACH_MAX_DURATION`, `BREACH_RECOVERY_SPEED`.
  - **Ticker:** `T` cycles `TICKERS` (SPY, QQQ, IWM, AAPL, TSLA); reloads price and history.
  - **Backtest:** `B` toggle; `←` / `→` step; `HOME` / `END` jump; optional auto-play (`BACKTEST_AUTO_PLAY`). Price from `load_stock_data` + `get_historical_price`.
  - **Reset:** `R` — new flock, new dog, IV default, breach clear, spot back to base.
  - **Print zones:** `P` dumps gamma zones to console.

- **Config constants (top of file):** `N`, `PEN_*`, `STRIKE_SPACING`, `DOG_*`, `SHEEP_*`, `FLOCK_*`, `GRAZING_*`, `BOUNDARY_*`, `BREACH_*`, `IV_*`, `BACKTEST_*`.

- **Observed outputs:**
  - **On-screen HUD:** Ticker, spot, IV%; breach state; total / grazing / in-pen counts; gamma flip and price range; backtest mode tag.
  - **Altimeter:** Horizontal bar mapping price range; spot and gamma-flip markers.
  - **Backtest slider:** Date and index when in backtest mode.
  - **Console:** Zone dump (`P`), breach messages, backtest mode, ticker switch, etc.
  - **CSV Logs:** Automatically saves run data to `logs/sheep_herding_stocks_log.csv` or `logs/sheep_herding_options_log.csv`.

- **No explicit PnL, score, or performance metric.** “Success” is implicit (sheep in pen, grazing, breach recovery).

---

## 5. DIFFERENCES BETWEEN THE TWO SCRIPTS

- **Structurally:** Same codebase. Only difference: first-line comment (stocks: `# sheep_herding_options`; options: `# sleep_herding_options` typo).

- **Entities, rules, parameters:** Identical. Both use gamma zones, IV, breach (put/call), flocking, grazing, pen, dog herding, backtest.

- **Interpretation:**
  - **Options version:** Directly reflects options-market concepts — gamma flip, put/call walls, IV scaling dealer reactivity, strike spacing, breach as move through strikes.
  - **Stocks version:** Same mechanics; can be interpreted as equity-only (retail flow, dealer containment, “price range”) without emphasizing options. No extra logic for payoff asymmetry, vanna, or charm; that would be future extension.

---

## 6. LIMITATIONS & ROUGH EDGES

1. **Logging is frame-based:** While CSV logging is now enabled and includes wall-clock timestamps, it logs every N frames (default 5). This captures high-frequency state but can generate large files over long sessions.

2. **No realistic time scaling:** Frame-based (60 fps). No explicit “1 frame = N seconds” or mapping to real trading intervals; backtest steps are “per day” (OHLC) but simulation time is uncoupled.

3. **Spot vs sheep centroid:** Spot is user/history-driven; sheep centroid is emergent. There’s no enforced coupling (e.g. spot = centroid, or centroid drives spot), so the “price = herd location” link is visual/metaphorical rather than mechanistic.

4. **Gamma zones are heuristic:** Flip and walls are computed from spot and `STRIKE_SPACING` (e.g. round to strike, ±2.5 strikes). No actual OI, gamma, or vanna inputs — limits fidelity for “flight envelope / Markov / gamma” integration.

5. **Single dog, homogeneous sheep:** One dealer, one flock type. No multiple MMs, no distinct retail cohorts (e.g. fast vs slow), no explicit order flow or volume.

6. **Breach is user-triggered:** Breaches are manual (L/S). No auto-breach from volatility, gamma exhaustion, or flow imbalance — reduces usefulness for exploring endogenous ruptures.

---

## 7. HANDOFF NOTES FOR CHATGPT

### Must-know summary

- **Sheep = retail flow,** **dog = dealer,** **pen = gamma-neutral zone,** **grassy patch = spot.** Screen X ↔ price via gamma ranges.
- **Dog** uses PD control + IV-scaled speed; **herds** by repulsion + “greener pastures” push toward pen. It **never enters the pen**; it positions between grazing cluster and pen.
- **Sheep** dynamics: flocking (separation, alignment, cohesion) + grazing (attract to spot, settle, wander) + dog repulsion + boundary repulsion. **Grazing count** and **in-pen count** are the main flow proxies.
- **Breach** = discrete L/S move past gamma flip, then **mean-reversion** to base flip. **IV** only affects dog agility, not sheep or zone geometry.
- **Stocks vs options:** Same codebase; options framing is explicit (gamma, IV, walls). Stocks variant is same mechanics, equity-oriented interpretation.

### Extension hooks

- **Gamma overlays:** Feed real gamma/OI (or flight-envelope outputs) into `calculate_gamma_zones` / zone drawing (e.g. flip, walls from data).
- **Herding-pressure logging:** Log dog target, sheep centroid, grazing/in-pen counts, breach events, and (if added) “spot vs centroid” each frame to CSV for later analysis.
- **Zone → put/call walls:** Replace heuristic walls with strike-level OI or delta-based levels; optionally drive pen geometry from same.
- **Endogenous breach:** Trigger breach from a volatility regime, gamma regime, or flow-imbalance metric instead of (or in addition to) L/S keys.
- **Time scaling:** Introduce “sim seconds per frame” and align backtest step length with simulation time; optionally sync to real-time or replay.

### Suggested questions for ChatGPT

1. **Logging Analysis:** Now that logging is active, how can we best visualize the relationship between the `datetime` wall-clock time and the simulated `frame_count` to detect lag or performance issues?
2. **Gamma parameterization:** How to replace heuristic flip/walls with inputs from a flight-envelope or gamma/OI layer (e.g. SPY envelope states, OI by strike) while keeping the same zone UI?
3. **SPY flight-envelope integration:** What’s a clean way to drive “regime” (e.g. neg/pos gamma, breach propensity) from `flight_envelope` or Markov states, and surface it in the sim (e.g. zone coloring, dog behavior)?
4. **Endogenous breaches:** How could breach triggers be derived from simulated volatility, dealer gamma exposure, or aggregate flow (e.g. centroid far from pen, grazing spike) instead of only L/S keys?
5. **Time alignment:** How to define “1 sim second = N frames” and optionally tie backtest step length (e.g. 1 day) to sim time so that breach duration, recovery speed, etc. can be interpreted in real time?

---

*End of summary. Paste this into ChatGPT when discussing changes, extensions, or integration with flight envelope / Markov / gamma tooling.*
