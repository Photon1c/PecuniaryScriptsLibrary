# BSM + Aerotrader Quickstart

## 🚀 Getting Started

### Prerequisites

1. **Python 3.8+** with required packages:
   ```bash
   pip install pandas pydantic
   ```

2. **Stock and Options Data**: Configure data paths in `aerotrader/modular/settings.json`

### Installation

No installation needed! The integration is ready to use.

## 🎯 Basic Commands

### 1. Generate a Single Trading Plan

Analyze a stock and generate a trading plan:

```bash
python cli.py plan --symbol SPY
```

Output: Trading plan JSON + formatted report

### 2. Continuous Monitoring

Watch a symbol and generate plans when signals appear:

```bash
python cli.py watch --symbol AAPL
```

The watcher will:
- Poll every 15 seconds (configurable in `config.py`)
- Run flight simulation
- Generate signals
- Create trading plans when opportunities are found

### 3. Intraday Mode

Use intraday simulation for finer granularity:

```bash
# One-shot plan with intraday
python cli.py plan --symbol TSLA --flight-mode intraday

# Watch with intraday
python cli.py watch --symbol NVDA --flight-mode intraday
```

## 📊 Understanding the Output

### Flight Telemetry Example

```
[BSM] SPY | Altitude: +2.5% | Phase: Thrust | Fuel: 75.0% | Stalls: 0
```

- **Altitude**: Net gain/loss percentage
- **Phase**: Flight phase (Thrust/Stall/Hover/Go-around)
- **Fuel**: Liquidity/momentum indicator
- **Stalls**: Risk events detected

### Signal Example

When a signal is generated:

```
[BSM] ✈️ Plan @ SPY -> flight_plan_2024-01-15.json
```

The signal includes:
- **Direction**: CALL or PUT
- **Confidence**: 0.0 to 1.0
- **Flight metrics**: Net gain, stalls, phase, fuel
- **Turbulence score**: Volatility assessment

## 🔧 Configuration

### Adjust Polling Interval

Edit `config.py`:

```python
POLL_SECS = 30  # Poll every 30 seconds instead of 15
```

### Change Signal Thresholds

Edit `signals.py` to customize when signals are generated:

```python
# Bullish conditions
is_bullish = (
    fd.net_gain > 1.0 and      # Require at least 1% gain
    fd.stall_events == 0 and   # No stalls allowed
    fd.fuel_remaining > 50     # At least 50% fuel
)

# Bearish conditions
is_bearish = (
    fd.net_gain < -1.5 or      # More than 1.5% loss
    fd.stall_events >= 3       # 3+ stall events
)
```

## 🧪 Testing the Integration

Run the test suite to verify everything works:

```bash
python test_integration.py
```

This will test:
1. Aerotrader JSON output
2. Schema validation
3. Ingest function
4. Signal computation

## 📈 Example Workflow

### Daily Trading Setup

1. **Morning**: Run a plan for your watchlist
   ```bash
   python cli.py plan --symbol SPY
   python cli.py plan --symbol QQQ
   python cli.py plan --symbol IWM
   ```

2. **During Market Hours**: Watch key positions
   ```bash
   python cli.py watch --symbol AAPL
   ```

3. **Review**: Check generated plans in output files

### Intraday Scalping Setup

Monitor with intraday mode for faster signals:

```bash
python cli.py watch --symbol SPY --flight-mode intraday
```

## 🐛 Troubleshooting

### "No module named 'core'"

Make sure aerotrader is run from the correct directory. The integration handles this automatically, but if you run aerotrader directly:

```bash
cd aerotrader/modular
python entry.py --symbol SPY
```

### "No data for ticker"

Check your data files in `aerotrader/modular/settings.json`:

```json
{
  "stock_data_dir": "path/to/your/stock/data",
  "option_data_dir": "path/to/your/option/data"
}
```

### No signals generated

This is normal! Signals are only generated when conditions are met:
- Bullish: Positive gain, low stalls, good fuel
- Bearish: Negative gain or high stalls

Try different symbols or wait for market conditions to change.

## 📚 Next Steps

- Read the [Integration Guide](INTEGRATION_GUIDE.md) for detailed architecture
- Customize signal logic in `signals.py`
- Implement LLM reasoning in `planner.py`
- Add risk rules in `risk_officer.py`

## 💡 Tips

1. **Start with daily mode** - It's faster and good for position trading
2. **Use intraday for scalping** - More granular but requires more data
3. **Adjust POLL_SECS** - Lower for faster updates, higher to save resources
4. **Monitor fuel levels** - Low fuel often precedes reversals
5. **Watch for phase changes** - Thrust→Stall transitions signal caution

---

Happy trading! ✈️

