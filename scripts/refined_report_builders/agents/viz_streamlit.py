import json
from pathlib import Path
import re
import streamlit as st

TRANSCRIPTS_DIR = (Path(__file__).parent / "transcripts").resolve()

def latest_transcript() -> Path:
    try:
        files = sorted(
            TRANSCRIPTS_DIR.glob("*_tinytroupe_v1_2.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if files:
            return files[0]
    except Exception:
        pass
    return Path("transcripts/tinytroupe_v1_2.jsonl")

def load_rows(p: Path):
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def to_items(raw):
    def classify(text: str) -> str:
        t = (text or "").lower()
        neg = ["risk", "caution", "concern", "uncertain", "bearish", "invalid", "stop-loss", "warning", "hedge"]
        pos = ["agree", "support", "bullish", "improve", "favorable", "opportunity", "confidence", "constructive"]
        score = sum(1 for w in pos if w in t) - sum(1 for w in neg if w in t)
        return "positive" if score > 0 else ("negative" if score < 0 else "neutral")

    items = []
    for row in raw:
        if isinstance(row, dict) and isinstance(row.get("turn"), dict):
            turn = row["turn"]
            rnd = row.get("round")
            step = f"Round {rnd}" if isinstance(rnd, int) else (str(rnd) if rnd is not None else "Unknown step")
            agent = turn.get("name") or turn.get("agent") or "Unknown agent"
            act = turn.get("role") or turn.get("act") or "TALK"
            content = turn.get("content") or turn.get("text") or ""
        else:
            step = row.get("step") or row.get("scene") or "Unknown step"
            agent = row.get("agent") or row.get("speaker") or "Unknown agent"
            act = row.get("act") or row.get("action") or "TALK"
            content = row.get("content") or row.get("text") or ""
        items.append({
            "step": step,
            "agent": agent,
            "act": act,
            "content": content.strip(),
            "sentiment": classify(content),
            "raw": row,
        })
    return items

def color_for(sentiment: str) -> str:
    return {
        "positive": "#16a34a",
        "negative": "#dc2626",
        "neutral": "#64748b",
    }.get(sentiment or "neutral", "#64748b")

def main():
    st.set_page_config(page_title="Agent Reasoning Dashboard", layout="wide")
    st.title("🛰️ Agent Reasoning Dashboard (Streamlit)")

    path = latest_transcript()
    st.caption(f"Using: {path}")
    raw = load_rows(path)
    items = to_items(raw)

    # Sidebar filters
    st.sidebar.header("Filters")
    q = st.sidebar.text_input("Keyword (agent or content)")
    agent_whitelist = sorted({it["agent"] for it in items})
    selected_agents = st.sidebar.multiselect("Agents", agent_whitelist, default=agent_whitelist)
    sentiments = st.sidebar.multiselect("Sentiment", ["positive", "neutral", "negative"], default=["positive", "neutral", "negative"])

    if q:
        ql = q.lower()
        items = [it for it in items if ql in (it["agent"].lower() + " " + it["content"].lower())]
    items = [it for it in items if it["agent"] in selected_agents and it["sentiment"] in sentiments]

    # Group by step
    grouped = {}
    for it in items:
        grouped.setdefault(it["step"], []).append(it)

    for step, rows in grouped.items():
        with st.expander(f"{step} — {len(rows)} turns", expanded=False):
            for it in rows:
                col1, col2 = st.columns([1, 9])
                with col1:
                    st.markdown(f"<div style='background:{color_for(it['sentiment'])};color:white;padding:4px 8px;border-radius:6px;font-size:12px;text-transform:uppercase'>{it['agent']}</div>", unsafe_allow_html=True)
                    st.caption(it["act"]) 
                with col2:
                    st.markdown(f"<div style='font-size:14px;white-space:pre-wrap'>{it['content']}</div>")

if __name__ == "__main__":
    main()


