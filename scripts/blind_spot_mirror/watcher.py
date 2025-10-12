import time
from ingest import run_aerotrader
from signals import compute_signals
from planner import reason_play
from risk_officer import enforce
from scribe import write_jsonl, write_flight_plan
from config import POLL_SECS

def watch(symbol: str = "SPY", flight_mode: str = "daily"):
    """
    Continuously monitor a symbol using aerotrader flight simulation.
    
    Args:
        symbol: Stock ticker to monitor (default: "SPY")
        flight_mode: Simulation mode - "daily" or "intraday" (default: "daily")
    """
    print(f"[BSM] Starting watch on {symbol} (mode: {flight_mode})")
    
    while True:
        try:
            ss = run_aerotrader(symbol, flight_mode)
            sig = compute_signals(ss)
            
            # Log flight status
            if ss.flight_data:
                fd = ss.flight_data
                print(f"[BSM] {symbol} | Altitude: {fd.net_gain:+.2f}% | "
                      f"Phase: {fd.latest_phase} | Fuel: {fd.fuel_remaining:.1f}% | "
                      f"Stalls: {fd.stall_events}")
            
            if not sig.get("best"): 
                print(f"[BSM] No signals at {symbol}")
                time.sleep(POLL_SECS)
                continue
                
            card = reason_play(ss, sig)
            card = enforce(card)
            write_jsonl(card)
            fp = write_flight_plan(card)
            print(f"[BSM] ✈️ Plan @ {symbol} -> {fp}")
        except Exception as e:
            print(f"[BSM] ⚠️ Error: {e}")
        
        time.sleep(POLL_SECS)
