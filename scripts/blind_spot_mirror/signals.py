from typing import Dict, Any, List
from schemas import Snapshot

def compute_signals(ss: Snapshot) -> Dict[str, Any]:
    """
    Compute trading signals from snapshot data.
    
    Supports two modes:
    1. Options chain analysis (if ss.chain has data)
    2. Flight simulation analysis (if ss.flight_data is available)
    """
    signals = {
        "best": None,
        "count": 0,
        "source": "none"
    }
    
    # If we have options chain data, use traditional analysis
    if ss.chain and len(ss.chain) > 0:
        candidates = []
        for row in ss.chain:
            spread = max(0.0, (row.get("ask", 0) - row.get("bid", 0)))
            if 0.18 <= abs(row.get("delta", 0.25)) <= 0.35 and spread <= 0.05 and row.get("oi", 0) >= 300:
                rr = (row["ask"]*1.5 - row["ask"]) / max(0.01, row["ask"]-row["bid"])  # toy RR
                candidates.append((rr, row))
        best = sorted(candidates, key=lambda x: x[0], reverse=True)[:1]
        signals = {
            "best": best[0][1] if best else None,
            "count": len(candidates),
            "source": "options_chain"
        }
    
    # If we have flight data, use flight-based signals
    elif ss.flight_data:
        fd = ss.flight_data
        
        # Flight-based trading signals
        # Bullish: net gain positive, low stalls, not in stall phase, fuel sufficient
        is_bullish = (
            fd.net_gain > 0 and
            fd.stall_events <= 1 and
            fd.latest_phase not in ["Stall"] and
            fd.fuel_remaining > 30
        )
        
        # Bearish: negative gain, high stalls, or in stall phase
        is_bearish = (
            fd.net_gain < -2 or
            fd.stall_events >= 2 or
            fd.latest_phase == "Stall"
        )
        
        # Turbulence severity
        turbulence_score = fd.turbulence_heavy * 2 + fd.turbulence_moderate * 1
        
        # Build signal
        direction = "CALL" if is_bullish else ("PUT" if is_bearish else "NEUTRAL")
        confidence = 0.0
        
        if direction != "NEUTRAL":
            # Base confidence on multiple factors
            gain_factor = min(abs(fd.net_gain) / 10, 1.0)  # normalize to 0-1
            stall_factor = 1.0 - (fd.stall_events / 5)     # fewer stalls = higher confidence
            fuel_factor = fd.fuel_remaining / 100          # more fuel = higher confidence
            turb_factor = max(0, 1.0 - turbulence_score / 10)  # less turbulence = higher
            
            confidence = (gain_factor + stall_factor + fuel_factor + turb_factor) / 4
        
        signals = {
            "best": {
                "direction": direction,
                "confidence": round(confidence, 2),
                "net_gain": fd.net_gain,
                "stall_events": fd.stall_events,
                "phase": fd.latest_phase,
                "fuel": fd.fuel_remaining,
                "turbulence_score": turbulence_score
            } if direction != "NEUTRAL" else None,
            "count": 1 if direction != "NEUTRAL" else 0,
            "source": "flight_simulation"
        }
    
    return signals
