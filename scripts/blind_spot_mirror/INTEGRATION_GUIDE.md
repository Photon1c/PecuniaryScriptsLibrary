# Aerotrader Integration Guide

## Overview

The **Blind Spot Mirror (BSM)** system now integrates with the **Aerotrader** flight simulation module to generate trading signals based on flight telemetry metaphors. This integration provides a unique perspective on market movements using aviation-inspired analysis.

## Architecture

```
Blind Spot Mirror (BSM)
├── cli.py              → Main CLI entry point
├── watcher.py          → Continuous monitoring mode
├── ingest.py           → Aerotrader integration layer
├── signals.py          → Signal computation (flight + options)
├── schemas.py          → Data models (Snapshot, FlightData, etc.)
├── planner.py          → LLM-based play reasoning
├── risk_officer.py     → Risk enforcement
└── scribe.py           → Output formatting

Aerotrader Module
└── aerotrader/modular/
    ├── entry.py        → Entry point
    └── core/
        ├── flight_sim_engine.py  → Main simulation engine
        ├── data_loader.py        → Stock/option data loading
        ├── candle_interpreter.py → Flight phase detection
        └── ...                   → Other flight modules
```

## How It Works

### 1. Data Flow

```
Stock Data → Aerotrader → Flight Telemetry → BSM Signals → Trading Plan
```

### 2. Flight Telemetry Metrics

The integration captures these key flight metrics:

- **Altitude (Net Gain)**: Cumulative percentage gain/loss
- **Fuel**: Liquidity/momentum indicator
- **Stall Events**: Risk indicators (EMA drag, bearish candles)
- **Turbulence**: Volatility classification (Heavy/Moderate/Calm)
- **Flight Phases**: Thrust, Stall, Hover, Go-around

### 3. Signal Generation

The `signals.py` module analyzes flight data to generate trading signals:

**Bullish Signals** (CALL):
- Net gain > 0%
- ≤ 1 stall event
- Not in "Stall" phase
- Fuel remaining > 30%

**Bearish Signals** (PUT):
- Net gain < -2%
- ≥ 2 stall events
- In "Stall" phase

**Confidence Scoring**:
- Gain factor: Based on magnitude of net gain
- Stall factor: Fewer stalls = higher confidence
- Fuel factor: More fuel = higher confidence
- Turbulence factor: Less turbulence = higher confidence

## Usage

### One-Shot Planning Mode

Generate a single trading plan for a symbol:

```bash
python cli.py plan --symbol SPY
```

With intraday simulation:

```bash
python cli.py plan --symbol AAPL --flight-mode intraday
```

### Continuous Monitoring Mode

Watch a symbol continuously and generate plans when signals appear:

```bash
python cli.py watch --symbol SPY
```

With intraday mode:

```bash
python cli.py watch --symbol TSLA --flight-mode intraday
```

### Direct Aerotrader Invocation

You can also call aerotrader directly for testing:

```bash
# File output (traditional)
cd aerotrader/modular
python entry.py --symbol SPY --mode daily

# JSON to stdout (for pipeline)
python entry.py --symbol SPY --mode daily --output stdout
```

## Configuration

### config.py

```python
AEROTRADER_DIR = "path/to/aerotrader/modular"
AEROTRADER_CMD = [sys.executable, "entry.py"]
POLL_SECS = 15  # Seconds between watch cycles
MODEL = "gpt-4o-mini"  # LLM model for reasoning

THRESHOLDS = {
    "max_spread_abs": 0.10,
    "min_oi": 300,
    "ivjump_pp_5m": 8.0,
    "rr_min": 1.8
}
```

### Data Requirements

Aerotrader expects data files in the format specified in `aerotrader/modular/settings.json`:

- Stock data: CSV with Date, Open, High, Low, Close/Last, Volume
- Options data: CSV with expiration, strike, IV, OI, etc.

Update the paths in `settings.json` to point to your data directory.

## Data Schemas

### Snapshot (Enhanced)

```python
{
  "symbol": "SPY",
  "spot": 450.25,
  "date": "2024-01-15",
  "mode": "Macro Cruise (Daily)",
  "book": {
    "latest_close": 450.25,
    "latest_volume": 85000000
  },
  "chain": [],  # Optional: options chain data
  "flight_data": {
    "net_gain": 2.5,
    "max_altitude": 3.2,
    "fuel_remaining": 75.0,
    "stall_events": 0,
    "turbulence_heavy": 1,
    "turbulence_moderate": 2,
    "latest_phase": "Thrust",
    "telemetry": [
      {
        "time": "09:30",
        "altitude": 0.5,
        "fuel": 100.0,
        "stall": false,
        "turbulence": "Calm",
        "phase": "Thrust"
      },
      ...
    ]
  }
}
```

### Signal Output

```python
{
  "best": {
    "direction": "CALL",
    "confidence": 0.82,
    "net_gain": 2.5,
    "stall_events": 0,
    "phase": "Thrust",
    "fuel": 75.0,
    "turbulence_score": 4
  },
  "count": 1,
  "source": "flight_simulation"
}
```

## Extending the Integration

### Adding Custom Flight Signals

Edit `signals.py` to add your own signal logic:

```python
def compute_signals(ss: Snapshot) -> Dict[str, Any]:
    if ss.flight_data:
        fd = ss.flight_data
        
        # Your custom logic here
        is_breakout = (
            fd.latest_phase == "Thrust" and
            fd.net_gain > 5 and
            fd.turbulence_heavy == 0
        )
        
        if is_breakout:
            return {"best": {...}, "count": 1, "source": "custom"}
```

### Combining Flight + Options Analysis

The system supports hybrid analysis. If your snapshot has both `flight_data` and `chain` data, you can create combined signals:

```python
# In signals.py
if ss.flight_data and ss.chain:
    # Combine both analysis modes
    flight_signal = analyze_flight(ss.flight_data)
    options_signal = analyze_options(ss.chain)
    combined_signal = merge_signals(flight_signal, options_signal)
```

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError when running aerotrader**

Ensure you're in the correct directory:
```bash
cd aerotrader/modular
python entry.py --symbol SPY --output stdout
```

**2. No data for ticker**

Check that your data files exist and are correctly configured in `aerotrader/modular/settings.json`.

**3. JSON parsing errors**

Run aerotrader in file mode first to verify it works:
```bash
python entry.py --symbol SPY --mode daily
# Check the output in logs/
```

### Debug Mode

Set stderr to capture for debugging:

```python
# In ingest.py
out = subprocess.check_output(
    cmd, 
    text=True, 
    cwd=AEROTRADER_DIR,
    stderr=subprocess.STDOUT  # Capture all output
)
```

## Performance Notes

- **Daily mode**: Analyzes last 5 candles, fast execution
- **Intraday mode**: Simulates intraday path from one candle, slightly slower
- **Watch mode**: Polls every 15 seconds by default (configurable via `POLL_SECS`)

## Future Enhancements

Potential improvements:

1. **Streaming telemetry**: Real-time flight updates instead of polling
2. **Multi-symbol watch**: Monitor multiple tickers simultaneously
3. **Flight path visualization**: Chart altitude/fuel curves
4. **Adaptive thresholds**: Machine learning for signal parameters
5. **Backtesting framework**: Historical flight simulation analysis

## References

- [Aerotrader README](aerotrader/modular/README.md)
- [BSM Schemas](schemas.py)
- [Signal Logic](signals.py)

---

**Integration completed**: All placeholder aerotrader calls have been replaced with fully functional flight simulation integration. The system now bridges aviation metaphors with quantitative trading signals.

