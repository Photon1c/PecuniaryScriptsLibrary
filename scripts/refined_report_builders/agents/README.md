# AI Agent Discussion (v1.2)

This folder contains a small multi-agent discussion pipeline around market images (charts, gamma exposure, etc.), plus two lightweight dashboards for reviewing transcripts.

## Files at a glance

- `discuss_v12.py`: Core orchestration for v1.2.
  - Defines `AgentConfig` and `MultiImageDiscussionOrchestrator`.
  - Handles screenshots/uploads and normalizes inputs into image URLs.
  - Produces a JSONL transcript under `transcripts/` with turns like `{ round, turn: { name, role, content } }`.
- `run_v1.2.py`: Minimal example runner for the orchestrator.
  - Passes a small roster of agents and a list of pages/urls to analyze.
- `adapters/agent_backend.py`: Backend adapter (OpenAI) used by the orchestrator.
  - Translates text+image messages to the OpenAI chat.completions format.
- `basic.py`: Self-contained demo that screenshots a few pages, uploads to ImageKit, and asks a vision model to synthesize a summary which agents respond to.
- `viz.py`: Flask dashboard for viewing transcripts.
  - Auto-detects the latest `*_tinytroupe_v1_2.jsonl` and renders rounds/turns.
  - Supports keyword search (`?q=...`) and simple sentiment color-coding.
- `viz_streamlit.py`: Streamlit dashboard with richer filters (agent/sentiment search).
- `templates/dashboard.html`: Minimal Tailwind UI for the Flask dashboard.
- `transcripts/`: JSONL transcript output directory (auto-created by the orchestrator).
- `screenshots/`: Temporary screenshots used for upload.

## Requirements

- Python 3.10+
- Packages (install as needed):
  - `openai`, `python-dotenv`, `selenium`, `webdriver-manager`, `Pillow`, `imagekitio`
  - Dashboards: `Flask` (for `viz.py`) and `streamlit` (for `viz_streamlit.py`)

Environment variables:
- `OPENAI_API_KEY` (required)
- `IMAGEKIT_PRIVATE_KEY`, `IMAGEKIT_PUBLIC_KEY`, `IMAGEKIT_URL_ENDPOINT` (for uploads; optional if you pass direct image URLs)
- `HEADLESS` (optional, `1` by default; set `0` to show browser when screenshotting)

## Quick start

1) Run the v1.2 discussion

- Edit `run_v1.2.py` if desired (agents, topic, urls). Then:

```bash
python run_v1.2.py
```

- Output: a transcript JSONL is saved under `/transcripts/` (timestamped).

2) View in Flask dashboard

```bash
python viz.py
# open http://127.0.0.1:5000
# optional search: http://127.0.0.1:5000/?q=gamma
```

3) View in Streamlit dashboard

```bash
streamlit viz_streamlit.py
```

## Notes

- If you pass webpage URLs in `images`, the orchestrator will attempt to screenshot and upload them, then call a vision-capable model with proper image URLs.
- The OpenAI backend expects models that support images (e.g., `gpt-4o`, `gpt-4o-mini`).
- The sentiment tags are simple keyword heuristics meant only for quick visual triage.
