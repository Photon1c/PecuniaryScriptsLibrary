# blackbox.py

import json
import pandas as pd

def write_markdown_log(filepath, ticker, date, mode, gains, fuel, stalls, turbulence, timestamps, flight_phases):
    """
    Writes a markdown flight log with telemetry table and summary.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# ✈️ Flight Summary - {ticker} - {date}\n")
        f.write(f"- Mode: {mode}\n")
        f.write(f"- Net Gain: {gains[-1]:+.2f}%\n")
        f.write(f"- Max Altitude: {max(gains):+.2f}%\n")
        f.write(f"- Fuel Remaining: {fuel[-1]:.1f}%\n")
        f.write(f"- Stall Events: {sum(stalls)}\n")
        f.write(f"- Emergency Landings: 0\n\n")
        f.write("| Time  | Altitude (%) | Fuel (%) | Stall | Turbulence | Phase       | Status |\n")
        f.write("|-------|---------------|----------|--------|------------|-------------|--------|\n")
        for i in range(len(gains)):
            status = "Takeoff" if i == 0 else ("Landed" if i == len(gains)-1 else "Cruising")
            f.write(f"| {timestamps[i]} | {gains[i]:+.2f}% | {fuel[i]:.1f}% | {'Yes' if stalls[i] else 'No'} | {turbulence[i]} | {flight_phases[i]:8} | {status:7} |\n")


def write_json_log(filepath, ticker, date, mode, gains, fuel, stalls, turbulence, timestamps, flight_phases):
    """
    Writes a JSON flight log with all telemetry data.
    """
    log = {
        "ticker": ticker,
        "date": date,
        "mode": mode,
        "net_gain": gains[-1],
        "max_altitude": max(gains),
        "fuel_remaining": fuel[-1],
        "stall_events": int(sum(stalls)),
        "emergency_landings": 0,
        "telemetry": [
            {
                "time": timestamps[i],
                "altitude": gains[i],
                "fuel": fuel[i],
                "stall": bool(stalls[i]),
                "turbulence": turbulence[i],
                "phase": flight_phases[i],
                "status": ("Takeoff" if i == 0 else ("Landed" if i == len(gains)-1 else "Cruising"))
            } for i in range(len(gains))
        ]
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def write_log(filepath, ticker, date, mode, gains, fuel, stalls, turbulence, timestamps, flight_phases, format="markdown"):
    """
    Unified log writer. Set format to 'markdown' or 'json'.
    """
    if format == "json":
        write_json_log(filepath, ticker, date, mode, gains, fuel, stalls, turbulence, timestamps, flight_phases)
    else:
        write_markdown_log(filepath, ticker, date, mode, gains, fuel, stalls, turbulence, timestamps, flight_phases)
