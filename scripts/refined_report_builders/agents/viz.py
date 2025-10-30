from flask import Flask, render_template, jsonify, request
import json
from pathlib import Path

app = Flask(__name__)

# Resolve to the newest transcript matching *_tinytroupe_v1_2.jsonl
TRANSCRIPTS_DIR = (Path(__file__).parent / "transcripts").resolve()

def _latest_transcript_path() -> Path:
    try:
        candidates = sorted(
            TRANSCRIPTS_DIR.glob("*_tinytroupe_v1_2.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    except Exception:
        pass
    # fallback to legacy default location/name
    return Path("transcripts/tinytroupe_v1_2.jsonl")

DATA_PATH = _latest_transcript_path()


def load_runs():
    """Load agent turns from JSONL and normalize them."""
    runs = []
    if not DATA_PATH.exists():
        return runs

    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            runs.append(obj)
    return runs


def to_dashboard_items(raw):
    """
    Shape the JSONL rows into:
    [
      {
        "step": "Trading Room step 4 of 4",
        "agent": "tiffany",
        "role": "TALK",
        "content": "...",
        "meta": {...}
      },
      ...
    ]
    """
    items = []
    def classify_sentiment(text: str) -> str:
        t = (text or "").lower()
        negative_cues = ["risk", "caution", "concern", "uncertain", "bearish", "drawdown", "invalid", "stop-loss", "volatility", "spike", "downside", "warning", "hedge"]
        positive_cues = ["agree", "support", "bullish", "improve", "favorable", "upside", "opportunity", "confidence", "constructive"]
        score = 0
        for w in negative_cues:
            if w in t:
                score -= 1
        for w in positive_cues:
            if w in t:
                score += 1
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"
    for row in raw:
        # Handle v1.2 transcript rows shaped like {"round": X|"final", "turn": {"role", "name", "content"}}
        if isinstance(row, dict) and isinstance(row.get("turn"), dict):
            turn = row["turn"]
            rnd = row.get("round")
            if isinstance(rnd, int):
                step = f"Round {rnd}"
            else:
                step = str(rnd) if rnd is not None else "Unknown step"

            agent = turn.get("name") or turn.get("agent") or "Unknown agent"
            act = turn.get("role") or turn.get("act") or "TALK"
            content = turn.get("content") or turn.get("text") or ""
        else:
            # Fallback for other historical shapes
            step = row.get("step") or row.get("scene") or "Unknown step"
            agent = row.get("agent") or row.get("speaker") or "Unknown agent"
            act = row.get("act") or row.get("action") or "TALK"
            content = row.get("content") or row.get("text") or ""

        items.append(
            {
                "step": step,
                "agent": agent,
                "act": act,
                "content": (content or "").strip(),
                "sentiment": classify_sentiment(content),
                "raw": row,
            }
        )
    return items


@app.route("/")
def index():
    raw = load_runs()
    items = to_dashboard_items(raw)

    # Keyword filter
    q = (request.args.get("q") or "").strip().lower()
    if q:
        items = [it for it in items if q in (it["content"].lower() + " " + it["agent"].lower())]

    # group by step so the UI can accordion it
    grouped = {}
    for it in items:
        grouped.setdefault(it["step"], []).append(it)

    # sort steps by name to make it predictable
    ordered_steps = sorted(grouped.keys())
    return render_template("dashboard.html", steps=ordered_steps, grouped=grouped, q=q)


@app.route("/api/agents")
def api_agents():
    raw = load_runs()
    items = to_dashboard_items(raw)
    return jsonify(items)


if __name__ == "__main__":
    app.run(debug=True)
