import json
from schemas import Snapshot, PlayCard
from prompts import PLAN_PROMPT
from config import THRESHOLDS
# from openai import OpenAI  # your client; omitted here per token budget

def reason_play(snapshot: Snapshot, signals: dict) -> PlayCard:
    prompt = PLAN_PROMPT.format(**{
        "min_oi": THRESHOLDS["min_oi"],
        "max_spread_abs": THRESHOLDS["max_spread_abs"],
        "rr_min": THRESHOLDS["rr_min"]
    })
    payload = {"SNAPSHOT": snapshot.model_dump(), "SIGNALS": signals}
    # resp = client.chat.completions.create(model=MODEL, messages=[...])
    # content = resp.choices[0].message.content
    content = "{}"  # placeholder; wire to your LLM in your env
    data = json.loads(content)
    return PlayCard(**data)
