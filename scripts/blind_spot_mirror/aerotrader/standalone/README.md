# Standalone Flight Simulation Scripts

This directory contains experimental, legacy, and simplified versions of the Aerotrader flight simulation engine. Each script demonstrates a different approach or stage of development for simulating stock price movement as aircraft flight using daily OHLCV and options data.

These scripts are intended for reference, rapid prototyping, and learning. For the latest, fully modular engine, see the `modular/` directory.

## File Descriptions

- **flight_sim_enginev1.py**: The most basic version. Uses mock data and hardcoded logic to generate a simple markdown flight log. No real data loading or advanced features.

- **flight_sim_enginev2.py**: Adds support for loading real stock and option data using a local `data_loader.py`. Still uses mock/synthetic intraday times and metrics, but demonstrates integration with real data sources.

- **flight_sim_enginev3.py**: Upgrades v2 by generating synthetic intraday times for the most recent date in the stock data. Uses either real or mock data for the output table, depending on data availability. Output is saved to the `logs/` subdirectory.

- **flight_sim_enginev4.py**: Further refines v3 by improving the sampling logic, fuel calculation, and status assignment. Handles both daily and synthetic intraday output, and includes more nuanced status logic (e.g., "Go-around pattern").

- **data_loader.py**: Utility for loading stock and option data from CSV or other sources. Used by the engine scripts.

- **settings.json**: Stores configuration settings for the standalone scripts.

## Usage

Each script can be run directly as a standalone CLI tool:
```bash
python flight_sim_enginev3.py
```

Logs are saved to the `logs/` subdirectory within `standalone/`.

## Notes
- These scripts are not as robust or feature-complete as the modular engine, but are useful for experimentation and educational purposes.
- Check back for updated versions or new experimental scripts in the future. 