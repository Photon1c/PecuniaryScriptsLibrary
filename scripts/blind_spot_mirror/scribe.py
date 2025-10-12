from datetime import datetime
import json, os

def write_jsonl(card, path="reports/plays.jsonl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(card.model_dump())+"\n")

def write_flight_plan(card, path_dir="reports"):
    os.makedirs(path_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = os.path.join(path_dir, f"flight_plan_{card.symbol}_{ts}.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(f"# Flight Plan: {card.symbol}\n\n")
        
        # Add current price and data date for verification
        spot_price = card.audit.get("spot_price", "N/A")
        data_date = card.audit.get("date", "N/A")
        flight_mode = card.audit.get("flight_mode", "N/A")
        
        f.write(f"## Data Source Verification\n")
        f.write(f"- **Current Spot Price**: ${spot_price:.2f} (from CSV last row)\n")
        f.write(f"- **Data Date**: {data_date}\n")
        f.write(f"- **Flight Mode**: {flight_mode}\n")
        f.write(f"- **Generated**: {card.audit.get('generated_at', 'N/A')}\n\n")
        
        f.write(f"## Trade Plan\n")
        f.write(f"- **Direction**: {card.direction}\n")
        f.write(f"- **Confidence**: {card.confidence:.0%}\n")
        f.write(f"- **Horizon**: {card.horizon}\n")
        f.write(f"- **Entry**: {card.entry}\n")
        f.write(f"- **Stop**: {card.stop}\n")
        f.write(f"- **Targets**: {card.targets}\n")
        f.write(f"- **Risk/Reward**: {card.risk_reward:.2f}\n\n")
        
        f.write(f"## Flight Conditions\n")
        f.write(f"- **Preconditions**: {card.preconditions}\n")
        f.write(f"- **Alerts**: {card.alerts}\n\n")
        
        f.write(f"## Notes\n")
        f.write(f"{card.notes}\n")
    return p
