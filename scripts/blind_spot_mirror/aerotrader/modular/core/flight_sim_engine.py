# flight_sim_engine.py
from datetime import datetime
import pandas as pd
import time
import argparse
import os

from .data_loader import load_stock_data, load_option_data
from .flight_path import compute_altitude_series
from .fuel_gauge import compute_fuel_levels
from .stall_detector import detect_stalls
from .turbulence_sensor import detect_iv_turbulence
from .blackbox import write_log

# --- CLI Config ---
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["daily", "intraday"], default="daily", help="Flight simulation mode")
parser.add_argument("--ticker", type=str, default="SPY", help="Ticker symbol")
parser.add_argument("--date", type=str, help="Simulation date (YYYY-MM-DD)")
parser.add_argument("--log-format", type=str, default="markdown", choices=["markdown", "json"], help="Log output format")
parser.add_argument("--output", type=str, default="file", choices=["file", "stdout"], help="Output destination")
parser.add_argument("--symbol", type=str, help="Alias for --ticker")
args = parser.parse_args()

# Support both --ticker and --symbol
if args.symbol:
    args.ticker = args.symbol
MODE = args.mode
TICKER = args.ticker
LOG_FORMAT = args.log_format

if args.date:
    sim_date = pd.to_datetime(args.date)
else:
    sim_date = None

# Set logs directory to the existing /modular/logs
LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs'))
OUTPUT_PATH = os.path.join(LOGS_DIR, f"flight_log_{TICKER.lower()}.{ 'json' if LOG_FORMAT == 'json' else 'md'}")
OUTPUT_MODE = args.output

# --- ASCII Plane Takeoff Animation (Single Line) ---
# Skip animation if outputting to stdout
if OUTPUT_MODE == "file":
    plane = "✈️"
    runway_length = 35
    print("\nPreparing for takeoff...")
    for i in range(runway_length):
        print("\r" + " " * i + plane, end="", flush=True)
        time.sleep(0.12)
    print("\r" + " " * (runway_length + 2), end="\r")  # Clear the line after animation
    print("🛫 Takeoff complete!\n")

# --- Load Data ---
stock_df = load_stock_data(TICKER)
option_df = load_option_data(TICKER)

# --- Preprocess Stock Data ---
stock_df['Date'] = pd.to_datetime(stock_df['Date'])
stock_df = stock_df.sort_values('Date')
stock_df['Close/Last'] = pd.to_numeric(stock_df['Close/Last'], errors='coerce')
stock_df['Volume'] = pd.to_numeric(stock_df['Volume'], errors='coerce')

from .candle_interpreter import apply_interpretation

if MODE == "daily":
    # --- Select Sampled Dates (last 5) ---
    if sim_date is not None:
        sampled = stock_df[stock_df['Date'] == sim_date].copy()
        if sampled.empty:
            raise ValueError(f"No data for {TICKER} on {sim_date.date()}")
    else:
        sampled = stock_df.tail(5).copy()
    prices = sampled['Close/Last']
    volumes = sampled['Volume']
    iv_series = pd.to_numeric(option_df['IV'], errors='coerce').dropna()
    stalls = detect_stalls(prices, sampled, iv_series)
    turbulence = detect_iv_turbulence(iv_series, sampled)
    from .fuel_gauge import generate_intraday_fuel_curve
    fuel = generate_intraday_fuel_curve(len(sampled))
    # For daily, use close-to-close gain as "altitude"
    altitudes = (prices.pct_change().fillna(0).cumsum() * 100).tolist()
    timestamps = sampled['Date'].dt.strftime('%H:%M').tolist() if 'Date' in sampled else [str(i) for i in range(len(sampled))]
    flight_phases = apply_interpretation(sampled)['Flight Phase'].tolist()
    mode_label = "Macro Cruise (Daily)"

elif MODE == "intraday":
    # --- Use only the last candle or specified date for intraday emulation ---
    if sim_date is not None:
        sampled = stock_df[stock_df['Date'] == sim_date].copy()
        if sampled.empty:
            raise ValueError(f"No data for {TICKER} on {sim_date.date()}")
    else:
        sampled = stock_df.tail(1).copy()
    from .intraday_emulator import simulate_intraday_path
    from .microturbulence import estimate_intraday_iv
    from .fuel_gauge import generate_intraday_fuel_curve
    candle = sampled.iloc[0]
    intraday_flight = simulate_intraday_path(candle)  # returns dict
    timestamps = list(intraday_flight.keys())
    altitudes = list(intraday_flight.values())
    fuel = generate_intraday_fuel_curve(len(altitudes))
    stalls = [False for _ in altitudes]  # Placeholder: no EMA stalls in this mode yet
    turbulence = estimate_intraday_iv(option_df.tail(1))[:len(altitudes)]
    # Infer flight phases from the full candle
    flight_phases = [apply_interpretation(sampled).iloc[0]['Flight Phase']] * len(altitudes)
    mode_label = "Jet Flight (Intraday)"
else:
    raise ValueError(f"Unknown mode: {MODE}")

# --- Verbose Output ---
if OUTPUT_MODE == "file":
    print("\nTurbulence Profile:")
    print(turbulence)
    if isinstance(turbulence, list) and len(turbulence) > 0 and isinstance(turbulence[0], str):
        print(f"Heavy: {turbulence.count('Heavy')}, Moderate: {turbulence.count('Moderate')}, Calm: {turbulence.count('Calm')}")

    print("\nStall Events:")
    print(stalls)
    print(f"Total Stalls: {sum(stalls)} / {len(stalls)}")

# --- Output Results ---
if OUTPUT_MODE == "stdout":
    # Output compact JSON to stdout for pipeline integration
    import json
    import sys
    output_data = {
        "symbol": TICKER,
        "date": (sim_date.strftime("%Y-%m-%d") if sim_date is not None else datetime.today().strftime("%Y-%m-%d")),
        "mode": mode_label,
        "spot": float(sampled['Close/Last'].iloc[-1]) if not sampled.empty else 0.0,
        "flight_data": {
            "net_gain": float(altitudes[-1]),
            "max_altitude": float(max(altitudes)),
            "fuel_remaining": float(fuel[-1]),
            "stall_events": int(sum(stalls)),
            "turbulence_heavy": turbulence.count('Heavy') if isinstance(turbulence, list) else 0,
            "turbulence_moderate": turbulence.count('Moderate') if isinstance(turbulence, list) else 0,
            "latest_phase": flight_phases[-1] if flight_phases else "Unknown",
            "telemetry": [
                {
                    "time": timestamps[i],
                    "altitude": float(altitudes[i]),
                    "fuel": float(fuel[i]),
                    "stall": bool(stalls[i]),
                    "turbulence": turbulence[i] if i < len(turbulence) else "Unknown",
                    "phase": flight_phases[i]
                } for i in range(len(altitudes))
            ]
        },
        "book": {
            "latest_close": float(sampled['Close/Last'].iloc[-1]) if not sampled.empty else 0.0,
            "latest_volume": int(sampled['Volume'].iloc[-1]) if not sampled.empty else 0
        },
        "chain": []  # Placeholder for options chain compatibility
    }
    json.dump(output_data, sys.stdout, indent=2)
else:
    # --- Write to File ---
    write_log(
        filepath=OUTPUT_PATH,
        ticker=TICKER,
        date=(sim_date.strftime("%m/%d/%Y") if sim_date is not None else datetime.today().strftime("%m/%d/%Y")),
        mode=mode_label,
        gains=altitudes,
        fuel=fuel,
        stalls=stalls,
        turbulence=turbulence,
        timestamps=timestamps,
        flight_phases=flight_phases,
        format=LOG_FORMAT
    )

    print(f"[✓] Flight log saved to {OUTPUT_PATH}")
    print(f"🔗 Sample Log: [`{OUTPUT_PATH}`](./{OUTPUT_PATH})")

    # --- Display Candle Interpreter Table as Markdown ---
    if MODE == "daily":
        interpreted_table = apply_interpretation(sampled)
        print("\n🕯️ Candle Phase Summary:")
        print("| Date       | Close    | Phase     |")
        print("|------------|----------|-----------|")
        for _, row in interpreted_table.iterrows():
            print(f"| {row['Date'].strftime('%Y-%m-%d')} | {row['Close/Last']:.2f}   | {row['Flight Phase']:<9} |")
    elif MODE == "intraday":
        print("\n🕯️ Candle Phase Summary:")
        print("| Time   | Altitude (%) | Phase     |")
        print("|--------|--------------|-----------|")
        for i, t in enumerate(timestamps):
            print(f"| {t} | {altitudes[i]:+.2f}       | {flight_phases[i]:<9} |")

