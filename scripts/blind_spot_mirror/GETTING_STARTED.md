# Getting Started with BSM + Aerotrader

## ✅ Pre-Integration Checklist

Before running the integrated system, ensure:

- [x] **Integration Complete**: All code files have been updated
- [x] **No Linter Errors**: All Python files pass linting
- [x] **Documentation**: README, guides, and examples created
- [ ] **Data Files**: Stock and option CSV files available
- [ ] **Configuration**: Paths configured in settings.json

## 🎯 Quick Start (5 Minutes)

### Step 1: Verify Installation

```bash
# Check Python version (3.8+ required)
python --version

# Install dependencies if needed
pip install pandas pydantic
```

### Step 2: Configure Data Paths

Edit `aerotrader/modular/settings.json`:

```json
{
  "stock_data_dir": "C:/path/to/your/stock/data",
  "option_data_dir": "C:/path/to/your/option/data"
}
```

**Data Format Requirements:**
- Stock CSV: `Date`, `Open`, `High`, `Low`, `Close/Last`, `Volume`
- Option CSV: Expiration, Strike, IV, OI, etc.

### Step 3: Test Aerotrader (Standalone)

```bash
cd aerotrader/modular
python entry.py --symbol SPY --mode daily
```

**Expected Output:**
- ✈️ Takeoff animation
- Turbulence profile
- Stall events count
- Log file created in `logs/`

### Step 4: Test Integration

```bash
# Return to project root
cd ../..

# Run test suite
python test_integration.py
```

**Expected Results:**
- ✅ Test 1: Aerotrader stdout JSON output
- ✅ Test 2: Snapshot schema validation
- ✅ Test 3: Ingest function
- ✅ Test 4: Signal computation

### Step 5: Run Your First Analysis

```bash
# One-shot analysis
python cli.py plan --symbol SPY
```

**What Happens:**
1. Loads SPY data via aerotrader
2. Runs flight simulation
3. Computes signals from flight data
4. Generates trading plan (if signal found)
5. Outputs plan to console and file

### Step 6: Start Monitoring

```bash
# Continuous watch mode
python cli.py watch --symbol SPY
```

**What You'll See:**
```
[BSM] Starting watch on SPY (mode: daily)
[BSM] SPY | Altitude: +2.5% | Phase: Thrust | Fuel: 75.0% | Stalls: 0
[BSM] ✈️ Plan @ SPY -> flight_plan_2024-01-15.json
```

Press `Ctrl+C` to stop.

## 🎓 Learning Path

### Beginner: Understanding the Output

1. **Read the telemetry**:
   - Altitude = price change %
   - Phase = market regime (Thrust/Stall/Hover/Go-around)
   - Fuel = momentum/liquidity
   - Stalls = risk events

2. **Interpret signals**:
   - CALL = bullish (positive gain, low stalls, good fuel)
   - PUT = bearish (negative gain, high stalls, stall phase)
   - Confidence = 0-1 score based on multiple factors

3. **Review plans**:
   - Entry parameters
   - Stop loss levels
   - Target prices
   - Risk/reward ratio

### Intermediate: Customization

1. **Adjust signal thresholds** in `signals.py`:
   ```python
   is_bullish = (
       fd.net_gain > 1.0 and      # Change from 0
       fd.stall_events <= 1 and
       fd.fuel_remaining > 50     # Change from 30
   )
   ```

2. **Change polling interval** in `config.py`:
   ```python
   POLL_SECS = 30  # Change from 15
   ```

3. **Add custom logging** in `watcher.py`:
   ```python
   if fd.turbulence_heavy > 3:
       print(f"⚠️ High turbulence warning!")
   ```

### Advanced: Extension

1. **Add new flight metrics** in `aerotrader/modular/core/flight_sim_engine.py`
2. **Implement LLM reasoning** in `planner.py`
3. **Connect broker API** in `scribe.py`
4. **Build backtesting framework** using historical data

## 🛠️ Common Workflows

### Daily Morning Routine

```bash
# Check your watchlist
python cli.py plan --symbol SPY
python cli.py plan --symbol QQQ
python cli.py plan --symbol AAPL
python cli.py plan --symbol TSLA

# Start watching your top pick
python cli.py watch --symbol SPY
```

### Intraday Scalping

```bash
# Use intraday mode for finer granularity
python cli.py watch --symbol SPY --flight-mode intraday
```

### Multi-Symbol Monitoring

Create a simple bash/PowerShell script:

```bash
# monitor.sh
for symbol in SPY QQQ IWM AAPL MSFT; do
  python cli.py plan --symbol $symbol >> daily_plans.log
done
```

```powershell
# monitor.ps1
$symbols = @("SPY", "QQQ", "IWM", "AAPL", "MSFT")
foreach ($symbol in $symbols) {
  python cli.py plan --symbol $symbol | Out-File -Append daily_plans.log
}
```

## 📊 Example Output Walkthrough

### 1. Console Output (Watch Mode)

```
[BSM] Starting watch on SPY (mode: daily)
[BSM] SPY | Altitude: +2.5% | Phase: Thrust | Fuel: 75.0% | Stalls: 0
```

**Interpretation:**
- Symbol moving up (+2.5%)
- In "Thrust" phase (strong momentum)
- Good fuel (75%)
- No stalls detected
- **Likely to generate bullish signal**

### 2. Signal Generated

```
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

**Interpretation:**
- Bullish (CALL) with 82% confidence
- Based on positive gain, no stalls, thrust phase
- Moderate turbulence (score 4)

### 3. Trading Plan

```
[BSM] ✈️ Plan @ SPY -> flight_plan_2024-01-15.json
```

Check `flight_plan_2024-01-15.json` for full details:
- Entry price/strike
- Stop loss
- Target prices
- Risk/reward ratio
- Preconditions and alerts

## 🐛 Troubleshooting

### "No module named 'core'"

**Problem:** Running aerotrader from wrong directory

**Solution:**
```bash
cd aerotrader/modular
python entry.py --symbol SPY
```

Or use the integration (handles this automatically):
```bash
python cli.py plan --symbol SPY
```

### "No data for ticker XXX"

**Problem:** Data files not found or ticker doesn't exist in your data

**Solution:**
1. Check `aerotrader/modular/settings.json` paths
2. Verify CSV files exist for that ticker
3. Try a ticker you know you have data for (e.g., SPY)

### "No signals generated"

**Problem:** Flight conditions don't meet signal thresholds

**Solution:** This is normal! Signals only generate when conditions are met:
- Try different symbols
- Wait for market conditions to change
- Lower thresholds in `signals.py` if testing

### JSON parsing error

**Problem:** Aerotrader output isn't valid JSON

**Solution:**
1. Test aerotrader standalone:
   ```bash
   cd aerotrader/modular
   python entry.py --symbol SPY --output stdout
   ```
2. Check for error messages in output
3. Verify data files are correctly formatted

## 📚 Documentation Reference

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Project overview |
| [QUICKSTART.md](QUICKSTART.md) | Quick command reference |
| [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) | Detailed integration docs |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture |
| [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md) | What was changed |
| This file | Step-by-step getting started |

## 🎉 You're Ready!

Once you've completed the checklist above, you're ready to:

✅ Generate trading signals from flight simulation data  
✅ Monitor symbols in real-time  
✅ Create automated trading plans  
✅ Extend the system with custom logic  

**Next Steps:**
1. Run the test suite: `python test_integration.py`
2. Try a simple plan: `python cli.py plan --symbol SPY`
3. Start watching: `python cli.py watch --symbol SPY`
4. Customize to your needs!

---

**Need Help?** Check the documentation or review the test suite for working examples.

