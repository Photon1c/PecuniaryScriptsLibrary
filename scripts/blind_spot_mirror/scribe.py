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
        f.write(f"- Horizon: {card.horizon}\n- Entry: {card.entry}\n- Stop: {card.stop}\n")
        f.write(f"- Targets: {card.targets}\n- RR: {card.risk_reward}\n")
        f.write(f"- Preconditions: {card.preconditions}\n- Alerts: {card.alerts}\n")
        f.write(f"- Notes: {card.notes}\n")
    return p
