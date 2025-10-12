# Modular Flight Simulation Engine

**Version:** 0.2.0

This folder contains the modular components of the Aerotrader flight simulation engine, which models stock price movement as aircraft flight using daily OHLCV and options data. The codebase is now organized for clarity and maintainability, with all core logic in `modular/core/` and a single entry point in `modular/entry.py`. The `standalone` directory contains multiple versions simplifying the modular version. `settings.json` is used to control the input directory paths for stock and option data retrieved from the Nasdaq and CBOE websites. all reports are saved to a `logs` directory.

## Directory Structure

- `modular/core/` — All core simulation modules and logic
- `modular/entry.py` — Main entry point for running the simulation
- `modular/generate_flight_report.py` — (if present) Utility for generating reports
- `modular/logs/` — All simulation log output (markdown and JSON)
- `modular/README.md` — This documentation

## Core Module Descriptions

- **flight_sim_engine.py**: Main simulation engine. Handles CLI, loads data, runs the simulation, and writes logs.
- **flight_ops_core.py**: Orchestrator for cross-domain simulation. Manages event triggers, state synchronization, and telemetry/history for market, aircraft, and traffic domains. Enables multi-domain and event-driven simulation scenarios.
- **blackbox.py**: Handles writing flight logs in markdown and JSON formats.
- **candle_interpreter.py**: Analyzes candle shapes and classifies them into flight phases (Thrust, Stall, Go-around, Hover).
- **data_loader.py**: Loads and preprocesses stock and option data from CSV or other sources.
- **flight_path.py**: Computes gain percentage per step for the simulated flight path.
- **fuel_gauge.py**: Models fuel (liquidity) consumption or decay during the simulation.
- **generate_flight_report.py**: Compiles the latest log, generates plots, and creates a printable Markdown report.
- **intraday_emulator.py**: Simulates synthetic intraday price paths from daily OHLC data.
- **microturbulence.py**: Estimates mock intraday IV (implied volatility) curves for turbulence modeling.
- **stall_detector.py**: Detects stall risk using EMA drag, candle shape, and IV delta.
- **turbulence_sensor.py**: Classifies turbulence for each step based on IV delta and candle shape.
- **settings.json**: Stores configuration settings for the simulation modules.

## Usage

### Running the Simulation
Always use the entry point:
```bash
python modular/entry.py [--mode daily|intraday] [--ticker TICKER] [--date YYYY-MM-DD] [--log-format markdown|json]
```
- `--mode` (`daily` or `intraday`): Select simulation mode (default: `daily`).
- `--ticker` (`str`): Ticker symbol to simulate (default: `SPY`).
- `--date` (`YYYY-MM-DD`): Simulate a specific date (must exist in your data).
- `--log-format` (`markdown` or `json`): Output log format (default: `markdown`).

**All logs are saved to `modular/logs/`.**

#### Example Commands
```bash
python modular/entry.py --ticker AAPL --mode intraday
python modular/entry.py --ticker TSLA --date 2024-06-25
python modular/entry.py --ticker MSFT --mode daily --log-format json
```

### Generating a Flight Report
If you have `generate_flight_report.py` in `modular/core/`, you can generate a Markdown report (with plots) from the latest log:
```bash
python modular/core/generate_flight_report.py
```

## Candle Phases Explained

The simulation uses flight metaphors to describe daily market behavior, derived from the shape of each candle:

- **Thrust:** Strong directional move (large body, small wicks). Indicates decisive market action or trend.
- **Go-around:** Large lower wick. Market dipped but recovered, signaling a failed descent or reversal attempt.
- **Hover:** Indecision or EMA drag zone (doji/small body). Market is pausing, consolidating, or showing uncertainty.
- **Stall:** Large upper wick or EMA drag. Market shows signs of exhaustion, rejection from highs, or risk of reversal.

These phases appear in the log tables to help interpret the market's "flight path" each day.

## Progress Tracking
- See `../development/progress.log` for a detailed changelog.
- Use `../development/progresstemplate.md` for new log entries.

## Unicode Compatibility
- All logs are written with UTF-8 encoding for full emoji and special character support on all platforms.

## Notes
- Do **not** run `flight_sim_engine.py` directly; always use `entry.py` for correct imports and CLI handling.
- The codebase is organized for easy extension and future modularization. 
- The standalone directory contains experimental scripts in fluid development, check back for their updated versions in the future.