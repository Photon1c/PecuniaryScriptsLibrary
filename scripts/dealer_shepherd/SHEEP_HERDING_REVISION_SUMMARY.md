# Sheep Herding — Minimal Revision Summary

## What changed

- **Logging (opt-in):** New config block `ENABLE_LOGGING`, `LOG_FILE_PATH`, `LOG_EVERY_N_FRAMES`. Helpers `init_logger()`, `log_frame(...)`, `compute_sheep_centroid(flock)`. On startup, if logging is on, the `logs` dir is created and the CSV is opened in append mode; header is written only when the file is new or empty. Each frame, when logging is on and `frame_count % LOG_EVERY_N_FRAMES == 0`, a row is appended. Logging is fully skipped when `ENABLE_LOGGING` is `False`.
- **Datetime timestamps:** Each log row now includes a `datetime` column with the actual wall-clock time in `YYYY-MM-DD HH:MM:SS` format, in addition to the frame-based `timestamp` column.
- **Separate log files:** The stocks script logs to `logs/sheep_herding_stocks_log.csv` and the options script logs to `logs/sheep_herding_options_log.csv` to avoid confusion between runs.
- **Logging enabled by default:** Both scripts now have `ENABLE_LOGGING = True` by default so logs are captured on every run.
- **Regime scaffold:** `RegimeState` enum (`RANGE_BOUND`, `BREACH_UP`, `BREACH_DOWN`). Main state holds `regime_state`; each frame it is set from `breach` (none → `RANGE_BOUND`, call → `BREACH_UP`, put → `BREACH_DOWN`). Used only for CSV logging and a new HUD line `Regime: RANGE-BOUND` / `BREACH-UP` / `BREACH-DOWN`. Dog/sheep behavior is unchanged.
- **Main loop:** `frame_count` (incremented each frame, reset on R), `regime_state` (updated after breach recovery, reset on R). `init_logger()` called once before the loop when logging is enabled; `log_frame(...)` called after sheep step when logging and log cadence allow.
- **Cleanups:** Top comment corrected to `# Sheep Herding - Stocks` and `# Sheep Herding - Options`. Both scripts share the same logging and regime logic. All existing controls (Space, L/S, R, +/-, T, P, B, arrows, HOME/END, ESC) unchanged.
- **CLI `--ticker`:** `argparse` added; `--ticker SPY` (or another ticker) overrides the config default. Use it to select the ticker at launch and ensure `data_loader` fetches the right price (e.g. SPY ~695). Run `python sheep_herding_stocks.py --help` for usage.
- **NaN-safe CSV:** `_csv_sanitize(v)` coerces logged values; `None` / `NaN` / `Inf` → `0` so the CSV has no NaN and stays parseable.
- **`--data-dir`:** Override `data_loader` base directory (folder containing `<TICKER>.csv`). Use when your data lives elsewhere than `F:/inputs/stocks`.

---

## How to enable logging and where the CSV appears

1. Set `ENABLE_LOGGING = True` in the config block at the top of `sheep_herding_stocks.py` or `sheep_herding_options.py`.
2. Run the script from any directory (e.g. project root or `metascripts/quickscripts`).
3. The CSV is always written to **`metascripts/quickscripts/logs/sheep_herding_log.csv`** (resolved relative to the script’s directory, not cwd). The `logs` folder is created if missing. With logging on, startup prints `Logging to <abs_path>`.

**Run with a specific ticker and data dir:**  
- `python sheep_herding_stocks.py --ticker SPY`  
- `python sheep_herding_stocks.py --ticker SPY --data-dir "D:/path/to/folder"`  

Use `--data-dir` when `data_loader` looks in the wrong place (default `F:/inputs/stocks`). The folder must contain `<TICKER>.csv`.

---

## CSV fields

| Column | Description |
|--------|-------------|
| `datetime` | Wall-clock time (`YYYY-MM-DD HH:MM:SS`). |
| `timestamp` | Sim time (frame count). |
| `ticker` | Current ticker (e.g. SPY). |
| `spot` | Spot price. |
| `gamma_flip` | Gamma flip level. |
| `put_wall` | Put wall price. |
| `call_wall` | Call wall price. |
| `iv` | IV %. |
| `breach_state` | `NONE` / `PUT` / `CALL`. |
| `n_sheep_total` | Total sheep. |
| `n_sheep_grazing_near_spot` | Sheep within grazing radius of spot. |
| `n_sheep_in_pen` | Sheep inside gamma-neutral pen. |
| `dog_x`, `dog_y` | Dog position. |
| `sheep_centroid_x`, `sheep_centroid_y` | Centroid of all sheep. |
| `regime_state` | `range_bound` / `breach_up` / `breach_down`. |

---

## Regime state: current behavior and extensions

**Current behavior:** `regime_state` is updated every frame from `breach` only: no breach → `RANGE_BOUND`; call breach → `BREACH_UP`; put breach → `BREACH_DOWN`. It is used for logging and the Regime HUD line. Dog and sheep logic do not read it.

**Extensions:** The scaffold can be extended by (1) driving `regime_state` from external inputs (e.g. flight-envelope or Markov states) instead of or in addition to `breach`; (2) adding more states (e.g. `NEAR_PUT_WALL`, `NEAR_CALL_WALL`); (3) using `regime_state` to modulate dog strength, sheep attraction, or zone rendering; (4) logging transition times between regimes for Markov inference or replay analysis.
