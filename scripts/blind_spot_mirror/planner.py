import json
from datetime import datetime, timedelta
from schemas import Snapshot, PlayCard
from prompts import PLAN_PROMPT
from config import THRESHOLDS
# from openai import OpenAI  # your client; omitted here per token budget

def reason_play(snapshot: Snapshot, signals: dict) -> PlayCard:
    """
    Generate a PlayCard from snapshot and signals.
    
    If LLM is not available, generates a basic PlayCard from flight data.
    To use LLM reasoning, uncomment the OpenAI code below.
    """
    
    # Option 1: Use LLM (commented out by default)
    # prompt = PLAN_PROMPT.format(**{
    #     "min_oi": THRESHOLDS["min_oi"],
    #     "max_spread_abs": THRESHOLDS["max_spread_abs"],
    #     "rr_min": THRESHOLDS["rr_min"]
    # })
    # payload = {"SNAPSHOT": snapshot.model_dump(), "SIGNALS": signals}
    # resp = client.chat.completions.create(model=MODEL, messages=[...])
    # content = resp.choices[0].message.content
    # data = json.loads(content)
    # return PlayCard(**data)
    
    # Option 2: Generate basic PlayCard from signals (default)
    best_signal = signals.get("best")
    if not best_signal:
        # No signal - create a neutral placeholder
        return PlayCard(
            symbol=snapshot.symbol,
            direction="NEUTRAL",
            horizon="1 day",
            entry={"price": snapshot.spot, "reason": "No signal"},
            stop={"price": snapshot.spot, "reason": "No position"},
            targets=[],
            risk_reward=0.0,
            confidence=0.0,
            preconditions=["No signal generated"],
            alerts=["No actionable signal at this time"],
            notes="No trading opportunity detected based on current flight metrics.",
            audit={
                "generated_at": datetime.now().isoformat(),
                "signal_source": signals.get("source", "none"),
                "signal_count": signals.get("count", 0)
            }
        )
    
    # Generate PlayCard from flight signal
    direction = best_signal.get("direction", "NEUTRAL")
    confidence = best_signal.get("confidence", 0.5)
    
    # Calculate entry, stop, and targets based on direction and current spot
    spot = snapshot.spot
    
    if direction == "CALL":
        # Bullish setup
        entry_price = spot * 1.02  # Enter slightly above current
        stop_price = spot * 0.98   # 2% stop loss
        targets = [spot * 1.05, spot * 1.08, spot * 1.10]  # 5%, 8%, 10% targets
        risk_reward = (targets[0] - entry_price) / (entry_price - stop_price)
        
        preconditions = [
            f"Net gain: {best_signal.get('net_gain', 0):+.2f}%",
            f"Phase: {best_signal.get('phase', 'Unknown')}",
            f"Stalls: {best_signal.get('stall_events', 0)}",
            f"Fuel: {best_signal.get('fuel', 0):.1f}%"
        ]
        
        alerts = [
            f"Monitor for stall phase entry",
            f"Watch for turbulence increase",
            f"Track fuel depletion below 30%"
        ]
        
        notes = (
            f"Bullish signal with {confidence:.0%} confidence. "
            f"Flight showing positive altitude ({best_signal.get('net_gain', 0):+.2f}%) "
            f"in {best_signal.get('phase', 'Unknown')} phase. "
            f"Low stall risk ({best_signal.get('stall_events', 0)} events). "
            f"Consider call options or long position."
        )
        
    elif direction == "PUT":
        # Bearish setup
        entry_price = spot * 0.98  # Enter slightly below current
        stop_price = spot * 1.02   # 2% stop loss
        targets = [spot * 0.95, spot * 0.92, spot * 0.90]  # -5%, -8%, -10% targets
        risk_reward = (entry_price - targets[0]) / (stop_price - entry_price)
        
        preconditions = [
            f"Net gain: {best_signal.get('net_gain', 0):+.2f}%",
            f"Phase: {best_signal.get('phase', 'Unknown')}",
            f"Stalls: {best_signal.get('stall_events', 0)}",
            f"Fuel: {best_signal.get('fuel', 0):.1f}%"
        ]
        
        alerts = [
            f"Monitor for phase reversal to Thrust",
            f"Watch for stall reduction",
            f"Track fuel recovery above 50%"
        ]
        
        notes = (
            f"Bearish signal with {confidence:.0%} confidence. "
            f"Flight showing negative altitude ({best_signal.get('net_gain', 0):+.2f}%) "
            f"or high stall risk ({best_signal.get('stall_events', 0)} events) "
            f"in {best_signal.get('phase', 'Unknown')} phase. "
            f"Consider put options or short position."
        )
        
    else:
        # Neutral - shouldn't happen but handle gracefully
        entry_price = spot
        stop_price = spot
        targets = []
        risk_reward = 0.0
        preconditions = ["Neutral signal"]
        alerts = ["No clear direction"]
        notes = "Neutral market conditions. Wait for clearer signal."
    
    horizon = "1-2 days" if snapshot.mode == "Macro Cruise (Daily)" else "intraday"
    
    return PlayCard(
        symbol=snapshot.symbol,
        direction=direction,
        horizon=horizon,
        entry={
            "price": round(entry_price, 2),
            "reason": f"Flight-based {direction} signal",
            "timing": "Market open" if direction != "NEUTRAL" else "Wait"
        },
        stop={
            "price": round(stop_price, 2),
            "reason": "2% risk limit",
            "type": "stop_loss"
        },
        targets=[round(t, 2) for t in targets],
        risk_reward=round(risk_reward, 2),
        confidence=round(confidence, 2),
        preconditions=preconditions,
        alerts=alerts,
        notes=notes,
        audit={
            "generated_at": datetime.now().isoformat(),
            "signal_source": signals.get("source", "unknown"),
            "signal_count": signals.get("count", 0),
            "flight_mode": snapshot.mode,
            "spot_price": snapshot.spot,
            "date": snapshot.date or datetime.now().strftime("%Y-%m-%d")
        }
    )
