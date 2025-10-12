# Blind Spot Mirror (BSM) 🛫

**Flight-based trading system powered by Aerotrader simulation engine**

BSM is a quantitative trading system that uses aviation metaphors to analyze market movements. By integrating with the Aerotrader flight simulation engine, it transforms stock price data into flight telemetry, generating trading signals based on altitude (gain), fuel (liquidity), stalls (risk events), and turbulence (volatility).

## ✨ Features

- **Flight Simulation Analysis**: Market movements analyzed as aircraft flight paths
- **Dual Mode Operation**: Daily (macro) and intraday (micro) analysis
- **Automated Signal Generation**: Bullish/bearish signals based on flight telemetry
- **Continuous Monitoring**: Watch mode for real-time signal detection
- **Risk-Aware**: Built-in stall detection and turbulence assessment
- **Extensible**: Easy to customize signal logic and thresholds

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│           Blind Spot Mirror (BSM)               │
├─────────────────────────────────────────────────┤
│  cli.py        → Main CLI interface             │
│  watcher.py    → Continuous monitoring          │
│  ingest.py     → Aerotrader integration         │
│  signals.py    → Signal computation             │
│  planner.py    → LLM-based reasoning            │
│  risk_officer  → Risk enforcement               │
│  scribe.py     → Output formatting              │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│              Aerotrader Engine                  │
├─────────────────────────────────────────────────┤
│  Flight Simulation → Telemetry Generation       │
│  • Altitude (gains)                             │
│  • Fuel (liquidity)                             │
│  • Stalls (risk events)                         │
│  • Turbulence (volatility)                      │
│  • Flight Phases (market regimes)               │
└─────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Generate a Trading Plan

```bash
python cli.py plan --symbol SPY
```

### 2. Watch a Symbol

```bash
python cli.py watch --symbol AAPL
```

### 3. Use Intraday Mode

```bash
python cli.py watch --symbol TSLA --flight-mode intraday
```

See [QUICKSTART.md](QUICKSTART.md) for detailed examples.

## 📊 How It Works

### Flight Metaphor → Trading Signals

| Flight Metric | Trading Interpretation | Signal Use |
|--------------|----------------------|-----------|
| **Altitude** | Net gain/loss % | Trend direction |
| **Fuel** | Liquidity/momentum | Sustainability |
| **Stalls** | Risk events | Warning signs |
| **Turbulence** | Volatility | Market conditions |
| **Phase** | Thrust/Stall/Hover | Regime detection |

### Signal Generation Logic

**Bullish (CALL):**
- ✅ Net gain > 0%
- ✅ Stall events ≤ 1
- ✅ Phase ≠ "Stall"
- ✅ Fuel > 30%

**Bearish (PUT):**
- ❌ Net gain < -2%
- ❌ Stall events ≥ 2
- ❌ Phase = "Stall"

**Confidence Score:**
Combines gain magnitude, stall frequency, fuel levels, and turbulence severity.

## 📁 Project Structure

```
blind_spot_mirror/
├── README.md                    # This file
├── QUICKSTART.md                # Quick start guide
├── INTEGRATION_GUIDE.md         # Detailed integration docs
├── test_integration.py          # Test suite
│
├── cli.py                       # Main CLI entry point
├── watcher.py                   # Continuous monitoring
├── ingest.py                    # Aerotrader integration
├── signals.py                   # Signal computation
├── planner.py                   # LLM reasoning
├── risk_officer.py              # Risk enforcement
├── scribe.py                    # Output formatting
├── schemas.py                   # Data models
├── config.py                    # Configuration
│
└── aerotrader/                  # Flight simulation engine
    └── modular/
        ├── entry.py             # Entry point
        └── core/                # Core modules
            ├── flight_sim_engine.py
            ├── data_loader.py
            ├── candle_interpreter.py
            └── ...
```

## 🔧 Configuration

### Basic Settings (config.py)

```python
POLL_SECS = 15                   # Watch mode polling interval
MODEL = "gpt-4o-mini"            # LLM for reasoning (optional)

THRESHOLDS = {
    "max_spread_abs": 0.10,      # Max option spread
    "min_oi": 300,               # Min open interest
    "rr_min": 1.8                # Min risk/reward ratio
}
```

### Data Setup

Configure stock and options data paths in `aerotrader/modular/settings.json`:

```json
{
  "stock_data_dir": "path/to/stock/data",
  "option_data_dir": "path/to/option/data"
}
```

## 🧪 Testing

Run the test suite to verify the integration:

```bash
python test_integration.py
```

Tests:
- ✅ Aerotrader stdout JSON output
- ✅ Schema validation
- ✅ Ingest function
- ✅ Signal computation

## 📖 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - Architecture and customization
- **[aerotrader/modular/README.md](aerotrader/modular/README.md)** - Aerotrader documentation

## 🎯 Use Cases

### Position Trading (Daily Mode)
- Monitor 5-day flight path
- Detect macro trend changes
- Generate swing trade signals

### Day Trading (Intraday Mode)
- Simulate intraday price action
- Detect micro stalls and reversals
- Generate scalping signals

### Portfolio Monitoring (Watch Mode)
- Continuous surveillance
- Alert on phase changes
- Auto-generate plans

## 🛠️ Extending BSM

### Custom Signal Logic

Edit `signals.py` to add your own indicators:

```python
def compute_signals(ss: Snapshot) -> Dict[str, Any]:
    if ss.flight_data:
        fd = ss.flight_data
        
        # Your custom logic
        is_breakout = (
            fd.latest_phase == "Thrust" and
            fd.net_gain > 5 and
            fd.fuel_remaining > 70
        )
        
        if is_breakout:
            return {"best": {...}, "source": "custom"}
```

### Hybrid Analysis

Combine flight data with traditional technical indicators or options flow.

## 📊 Example Output

### Watch Mode Console

```
[BSM] Starting watch on SPY (mode: daily)
[BSM] SPY | Altitude: +2.5% | Phase: Thrust | Fuel: 75.0% | Stalls: 0
[BSM] ✈️ Plan @ SPY -> flight_plan_2024-01-15.json

{
  "direction": "CALL",
  "confidence": 0.82,
  "net_gain": 2.5,
  "phase": "Thrust",
  "fuel": 75.0,
  "stall_events": 0
}
```

## 🤝 Integration with Other Systems

BSM's modular design allows easy integration:

- **LLM Reasoning**: Implement in `planner.py` for natural language trade plans
- **Risk Management**: Customize `risk_officer.py` for portfolio constraints
- **Execution**: Add broker API calls in `scribe.py`
- **Alerts**: Extend `watcher.py` with notifications (email, Slack, etc.)

## 🐛 Troubleshooting

See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md#troubleshooting) for common issues and solutions.

## 📝 License

This is a personal trading system. Use at your own risk. Not financial advice.

## 🙏 Acknowledgments

- **Aerotrader**: Flight simulation engine for market analysis
- Inspired by aviation safety principles applied to trading

---

**Ready to fly?** Start with [QUICKSTART.md](QUICKSTART.md) ✈️

