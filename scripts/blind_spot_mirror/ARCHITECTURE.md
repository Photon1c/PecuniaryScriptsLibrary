# BSM + Aerotrader Architecture

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                              │
├─────────────────────────────────────────────────────────────────────┤
│  cli.py                                                             │
│  • plan mode    → One-shot analysis                                 │
│  • watch mode   → Continuous monitoring                             │
│  • Arguments: --symbol, --flight-mode                               │
└──────────────────┬──────────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATION LAYER                            │
├─────────────────────────────────────────────────────────────────────┤
│  watcher.py (watch mode)                                            │
│  • Polling loop (configurable interval)                             │
│  • Live telemetry logging                                           │
│  • Error handling                                                   │
│  • Signal → Plan → Output pipeline                                  │
└──────────────────┬──────────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA INGESTION                                 │
├─────────────────────────────────────────────────────────────────────┤
│  ingest.py                                                          │
│  • run_aerotrader(symbol, mode)                                     │
│  • Subprocess management                                            │
│  • JSON parsing                                                     │
│  • Schema validation                                                │
└──────────────────┬──────────────────────────────────────────────────┘
                   │
                   │ subprocess call
                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   AEROTRADER ENGINE                                 │
├─────────────────────────────────────────────────────────────────────┤
│  aerotrader/modular/                                                │
│    entry.py → flight_sim_engine.py                                  │
│      ├── data_loader.py         (CSV reading)                       │
│      ├── candle_interpreter.py  (Phase detection)                   │
│      ├── stall_detector.py      (Risk events)                       │
│      ├── turbulence_sensor.py   (Volatility)                        │
│      ├── fuel_gauge.py          (Liquidity)                         │
│      └── intraday_emulator.py   (Intraday paths)                    │
│                                                                      │
│  Output: JSON to stdout                                             │
│    {                                                                 │
│      "symbol": "SPY",                                                │
│      "spot": 450.25,                                                 │
│      "flight_data": {...},                                           │
│      "book": {...},                                                  │
│      "chain": []                                                     │
│    }                                                                 │
└──────────────────┬──────────────────────────────────────────────────┘
                   │
                   │ Snapshot object
                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SIGNAL GENERATION                                │
├─────────────────────────────────────────────────────────────────────┤
│  signals.py                                                         │
│  • compute_signals(snapshot)                                        │
│  • Flight-based analysis:                                           │
│    - Bullish: gain>0, low stalls, good fuel                         │
│    - Bearish: gain<-2, high stalls, stall phase                     │
│  • Confidence scoring                                               │
│  • Options chain analysis (if available)                            │
│                                                                      │
│  Output: Signal dict                                                │
│    {                                                                 │
│      "best": {...},                                                  │
│      "count": 1,                                                     │
│      "source": "flight_simulation"                                   │
│    }                                                                 │
└──────────────────┬──────────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PLANNING LAYER                                  │
├─────────────────────────────────────────────────────────────────────┤
│  planner.py                                                         │
│  • reason_play(snapshot, signals)                                   │
│  • LLM-based reasoning (optional)                                   │
│  • Generate trade parameters                                        │
│  • Risk/reward calculation                                          │
└──────────────────┬──────────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     RISK MANAGEMENT                                 │
├─────────────────────────────────────────────────────────────────────┤
│  risk_officer.py                                                    │
│  • enforce(play_card)                                               │
│  • Validate against thresholds                                      │
│  • Position sizing                                                  │
│  • Portfolio constraints                                            │
└──────────────────┬──────────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        OUTPUT LAYER                                 │
├─────────────────────────────────────────────────────────────────────┤
│  scribe.py                                                          │
│  • write_jsonl(play_card)      → Structured logs                    │
│  • write_flight_plan(play_card) → Human-readable                    │
│  • Format for downstream systems                                    │
└─────────────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow: Single Request

```
1. User runs:  python cli.py plan --symbol SPY --flight-mode daily

2. CLI calls:  ss = run_aerotrader("SPY", "daily")

3. Ingest runs subprocess:
   [python, entry.py, --symbol, SPY, --mode, daily, --output, stdout]
   Working dir: aerotrader/modular/

4. Aerotrader:
   • Loads SPY stock data (CSV)
   • Loads SPY option data (CSV)
   • Computes flight telemetry
   • Outputs JSON to stdout

5. Ingest parses JSON → Snapshot object

6. Signals analyzes:
   Snapshot.flight_data → trading signal

7. Planner generates:
   Signal → PlayCard (entry, stop, targets)

8. Risk officer validates:
   PlayCard → Enforced PlayCard

9. Scribe writes:
   • JSONL log
   • Human-readable flight plan

10. Output displayed to user
```

## 🔄 Data Flow: Watch Mode

```
1. User runs:  python cli.py watch --symbol AAPL

2. Watcher starts infinite loop:

   LOOP:
     ├─→ run_aerotrader("AAPL", "daily")
     ├─→ compute_signals(snapshot)
     ├─→ Display live telemetry:
     │   [BSM] AAPL | Altitude: +1.2% | Phase: Thrust | Fuel: 80%
     ├─→ If signal detected:
     │   ├─→ reason_play(snapshot, signal)
     │   ├─→ enforce(play_card)
     │   ├─→ write_jsonl(play_card)
     │   └─→ write_flight_plan(play_card)
     ├─→ Sleep for POLL_SECS
     └─→ Repeat

3. User can Ctrl+C to stop
```

## 📊 Schema Flow

### Aerotrader Output → Snapshot

```python
# Aerotrader stdout JSON
{
  "symbol": "SPY",
  "date": "2024-01-15",
  "mode": "Macro Cruise (Daily)",
  "spot": 450.25,
  "flight_data": {
    "net_gain": 2.5,
    "max_altitude": 3.2,
    "fuel_remaining": 75.0,
    "stall_events": 0,
    "turbulence_heavy": 1,
    "turbulence_moderate": 2,
    "latest_phase": "Thrust",
    "telemetry": [...]
  },
  "book": {
    "latest_close": 450.25,
    "latest_volume": 85000000
  },
  "chain": []
}

# Validated as Snapshot
snapshot = Snapshot(**json_data)
```

### Snapshot → Signal

```python
# Input: Snapshot with flight_data
snapshot.flight_data = FlightData(
  net_gain=2.5,
  stall_events=0,
  latest_phase="Thrust",
  fuel_remaining=75.0,
  ...
)

# Output: Signal dict
signal = {
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

### Signal → PlayCard

```python
# Input: Snapshot + Signal
snapshot.symbol = "SPY"
snapshot.spot = 450.25
signal["best"]["direction"] = "CALL"

# Output: PlayCard (via planner.py)
play_card = PlayCard(
  symbol="SPY",
  direction="CALL",
  horizon="1-2 weeks",
  entry={...},
  stop={...},
  targets=[...],
  risk_reward=1.8,
  confidence=0.82,
  ...
)
```

## 🧩 Module Dependencies

```
cli.py
  └─→ ingest.py
  └─→ signals.py
  └─→ planner.py
  └─→ risk_officer.py
  └─→ scribe.py
  └─→ watcher.py

ingest.py
  └─→ schemas.py
  └─→ config.py
  └─→ subprocess (aerotrader)

signals.py
  └─→ schemas.py

schemas.py
  └─→ pydantic

config.py
  └─→ (no dependencies)

watcher.py
  └─→ ingest.py
  └─→ signals.py
  └─→ planner.py
  └─→ risk_officer.py
  └─→ scribe.py
  └─→ config.py
```

## 🔒 Error Handling

```
┌─────────────────────────────────────────────────────┐
│ Level 1: Subprocess Errors                          │
├─────────────────────────────────────────────────────┤
│ • Aerotrader process fails                          │
│ • Data files missing                                │
│ • Invalid ticker symbol                             │
│ → Caught in ingest.py subprocess.check_output()     │
│ → Raises FileNotFoundError or subprocess error      │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ Level 2: Data Validation Errors                     │
├─────────────────────────────────────────────────────┤
│ • Invalid JSON format                               │
│ • Schema validation failure                         │
│ • Missing required fields                           │
│ → Caught in ingest.py json.loads() or Snapshot()   │
│ → Raises JSONDecodeError or ValidationError        │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ Level 3: Signal Computation Errors                  │
├─────────────────────────────────────────────────────┤
│ • Missing flight_data or chain                      │
│ • Invalid signal parameters                         │
│ → Returns safe default: {"best": None, "count": 0} │
│ → No exception raised                               │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ Level 4: Watch Mode Errors                          │
├─────────────────────────────────────────────────────┤
│ • Any error in watch loop                           │
│ → Caught in try/except in watcher.py               │
│ → Logs error and continues                         │
│ → User can Ctrl+C to stop                          │
└─────────────────────────────────────────────────────┘
```

## 🎯 Extension Points

### 1. Custom Flight Indicators

Add new metrics to aerotrader:

```python
# In flight_sim_engine.py
output_data["flight_data"]["custom_metric"] = calculate_custom()

# In signals.py
if fd.custom_metric > threshold:
    # Generate custom signal
```

### 2. Alternative Signal Sources

Combine multiple data sources:

```python
# In signals.py
def compute_signals(ss: Snapshot) -> Dict:
    flight_signal = analyze_flight(ss.flight_data)
    options_signal = analyze_options(ss.chain)
    news_signal = analyze_news(ss.symbol)
    
    # Ensemble voting
    return combine_signals([flight_signal, options_signal, news_signal])
```

### 3. Real-time Data Integration

Replace CSV files with live data feeds:

```python
# In aerotrader/modular/core/data_loader.py
def load_stock_data(ticker):
    # Instead of CSV:
    # return pd.read_csv(...)
    
    # Use API:
    import yfinance as yf
    return yf.download(ticker, period="5d")
```

### 4. Execution Integration

Add broker API in scribe.py:

```python
# In scribe.py
def execute_play(play_card: PlayCard):
    if play_card.direction == "CALL":
        order = broker.buy_call(
            symbol=play_card.symbol,
            strike=play_card.entry["strike"],
            quantity=calculate_quantity(play_card)
        )
    # ...
```

## 📈 Performance Characteristics

| Component | Latency | Notes |
|-----------|---------|-------|
| Aerotrader (daily) | ~1-2s | 5 candles, lightweight |
| Aerotrader (intraday) | ~2-4s | Intraday emulation, more compute |
| Subprocess overhead | ~100-200ms | Python spawn time |
| Signal computation | <10ms | Pure computation |
| Watch loop total | ~2-5s | Per iteration |

**Optimization Tips:**
- Use daily mode for faster updates
- Increase POLL_SECS if latency is not critical
- Cache data files if running multiple times
- Consider running aerotrader as a service instead of subprocess

## 🔐 Security Considerations

1. **Subprocess Execution**: Uses `stderr=subprocess.DEVNULL` to prevent information leakage
2. **Working Directory**: Properly isolated to aerotrader module
3. **Data Validation**: All external data validated via Pydantic schemas
4. **No Shell Injection**: Uses list-based subprocess args, not shell strings

---

This architecture provides:
- ✅ Clean separation of concerns
- ✅ Easy to test each component
- ✅ Extensible for future enhancements
- ✅ Robust error handling
- ✅ Performance optimized for real-time use

