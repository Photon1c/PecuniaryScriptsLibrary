# AI Agent Discussion (v1.2)

A multi-agent discussion system for analyzing images (market charts, medical images, financial statements, etc.) with specialized roles, strict validation, and deterministic evaluation.

## What It Does

This system orchestrates conversations between AI agents with different roles (quant, risk manager, skeptic, radiologist, auditor, etc.) to analyze images and produce structured discussions. Key features:

- **Multi-agent discussions**: Agents with specialized roles analyze images together
- **Vision support**: Handles screenshots, web pages (auto-converts to images), and direct image URLs
- **Deterministic evaluation**: Eval harness produces diffable JSON for CI/CD
- **Strict validation**: Settings schema validation fails fast on configuration errors
- **Multiple dashboards**: Flask and Streamlit interfaces for viewing transcripts
- **Preset configurations**: Quick-start presets for common use cases
- **Offline testing**: Mock backend for testing without API calls

## Quick Start

### 1. Install Dependencies

```bash
pip install openai python-dotenv selenium webdriver-manager Pillow imagekitio
pip install flask streamlit jsonschema  # For dashboards and validation
```

### 2. Set Environment Variables

Create a `.env` file:

```bash
OPENAI_API_KEY=your_key_here
IMAGEKIT_PRIVATE_KEY=your_key  # Optional: only needed for image uploads
IMAGEKIT_PUBLIC_KEY=your_key
IMAGEKIT_URL_ENDPOINT=your_endpoint
HEADLESS=1  # Set to 0 to see browser during screenshots
```

### 3. Run a Discussion

**Option A: Using CLI with presets (recommended)**

```bash
# Quick start with default markets preset
python discuss_v12.py --preset markets

# Custom configuration
python discuss_v12.py --preset balanced --rounds 5 --topic "SPY volatility analysis" --backend openai

# Dry run to validate without executing
python discuss_v12.py --preset markets --dry-run
```

**Available presets:**
- `markets`: 3-agent setup (quant, risk_manager, skeptic) - **default**
- `minimal`: Single analyst
- `balanced`: Quant + Risk Manager
- `critical`: Two skeptics for critical analysis

**Option B: Direct Python script**

Edit `run_v1.2.py` to customize agents/images, then:

```bash
python run_v1.2.py
```

**Output:** Transcript saved to `transcripts/{timestamp}_tinytroupe_v1_2.jsonl`

### 4. View Results

**Flask Dashboard** (lightweight, keyword search):

```bash
python viz.py
# Open http://127.0.0.1:5000
# Search: http://127.0.0.1:5000/?q=gamma
```

**Streamlit Dashboard** (richer filters):

```bash
streamlit run viz_streamlit.py
# Opens in browser automatically
```

Both dashboards auto-detect the latest transcript and support sentiment color-coding.

### 5. Run Evaluations

The eval harness produces deterministic, diffable JSON for testing:

```bash
# Run all fixtures for a preset
python -m evals.run_agent_eval --preset markets --rounds 2 --backend openai --save

# Run specific fixture
python -m evals.run_agent_eval --fixture markets_1 --rounds 2 --save

# Offline mode (no API calls, mock backend)
python -m evals.run_agent_eval --preset markets --offline --save

# Custom seed for reproducibility (default: 42)
python -m evals.run_agent_eval --preset markets --seed 123 --save
```

**Output format:**
```json
{
  "fixture": "markets_1",
  "status": "ok",
  "backend": "openai",
  "rounds_completed": 2,
  "agents": ["Ava", "Blake", "Casey"],
  "token_usage": {"total_completion": 974},
  "latency": {"p50": 3.1, "p95": 6.8},
  "transcript": "transcripts/173032...jsonl",
  "timestamp": "2024-11-01T12:34:56.789Z"
}
```

Results saved to `evals/results/` when `--save` is used.

## Project Structure

### Core Files

- **`discuss_v12.py`**: Main orchestration engine
  - `AgentConfig`: Agent configuration dataclass
  - `MultiImageDiscussionOrchestrator`: Runs multi-agent discussions
  - Handles image normalization (screenshots → uploads → image URLs)
  - Tracks rounds, budget exhaustion, partial results
  - CLI with `--preset`, `--dry-run` support

- **`agent_loader.py`**: Agent construction from presets
  - Decouples agent creation from entrypoints
  - Loads presets from `settings.json` or built-in defaults
  - Returns `AgentConfig` objects with correct prompts
  - Used by `discuss_v12.py`, `run_v1.2.py`, and other entrypoints

- **`run_v1.2.py`**: Simple Python runner example

- **`adapters/agent_backend.py`**: Backend abstraction
  - `OpenAIBackend`: OpenAI implementation with vision support
  - Extensible for other providers (Anthropic, Google, etc.)

- **`adapters/prompts.py`**: Role and system prompts
  - Role templates: quant, risk_manager, skeptic, radiologist, auditor
  - Domain-specific notes: markets, medical, accounting

### Dashboards

- **`viz.py`**: Flask dashboard
  - Auto-detects latest transcript
  - Keyword search via URL parameter
  - Sentiment color-coding (green=positive, red=critical, gray=neutral)

- **`viz_streamlit.py`**: Streamlit dashboard
  - Multi-select filters (agents, sentiment)
  - Better UX for exploring large discussions

### Evaluation

- **`evals/run_agent_eval.py`**: Deterministic eval harness
  - Produces diffable JSON output
  - Captures metrics: latency, tokens, rounds, status
  - Supports `--offline` mode with mock backend
  - Handles partial results, missing tokens, vision support checks

- **`evals/fixtures/`**: Test fixtures
  - `markets_1.json`, `medical_1.json`, `accounting_1.json`
  - Add your own fixtures here

- **`evals/results/`**: Eval output (created when `--save` used)

### Configuration

- **`settings.schema.json`**: Strict JSON schema
  - Required: `transcripts_dir`, `default_backend`, `default_model`, `presets`
  - Rejects unknown keys (fails fast)

- **`settings.json.example`**: Example settings file

- **`settings_loader.py`**: Strict loader with validation
  ```python
  from settings_loader import load_settings
  cfg = load_settings("settings.json")  # Exits if validation fails
  ```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `IMAGEKIT_*` | No | ImageKit credentials (only if using uploads) |
| `HEADLESS` | No | Browser headless mode (`1`=yes, `0`=no, default: `1`) |
| `TT_SEED` | No | Random seed (default: 42) |
| `TT_MODEL` | No | Default model ID (default: `gpt-4o-mini`) |
| `TT_BACKEND` | No | Default backend (default: `openai`) |

### Settings File (Optional)

Create `settings.json` based on `settings.json.example`:

```json
{
  "transcripts_dir": "./transcripts",
  "default_backend": "openai",
  "default_model": "gpt-4o-mini",
  "presets": {
    "markets": {
      "topic_hint": "Market analysis",
      "agents": [
        {"name": "Ava", "role": "quant", "domain": "markets"},
        {"name": "Blake", "role": "risk_manager", "domain": "markets"},
        {"name": "Casey", "role": "skeptic", "domain": "markets"}
      ]
    }
  }
}
```

**Validation rules:**
- All required fields must be present
- Unknown keys are rejected (prevents typos)
- Exits with error if `jsonschema` not installed
- Does not auto-create directories if validation fails

## CLI Reference

### `discuss_v12.py`

```bash
python discuss_v12.py [OPTIONS]

Options:
  --preset {markets,minimal,balanced,critical}  Agent preset (default: markets)
  --images URL [URL ...]                        Image URLs/paths
  --backend BACKEND                              Backend name (default: openai)
  --rounds N                                     Number of rounds (default: 3)
  --seed N                                       Random seed (default: 42)
  --topic TEXT                                   Topic hint
  --dry-run                                      Validate config without executing
  --clear-env                                    Clear tinytroupe environment/agents before running
```

**Examples:**
```bash
# Quick start
python discuss_v12.py --preset markets

# Custom images
python discuss_v12.py --preset minimal --images https://example.com/chart.png

# Extended discussion
python discuss_v12.py --preset balanced --rounds 10 --topic "Deep analysis"

# Clear tinytroupe state before running (fixes "Agent name already in use" errors)
python discuss_v12.py --preset markets --clear-env
```

### `evals/run_agent_eval.py`

```bash
python -m evals.run_agent_eval [OPTIONS]

Options:
  --preset NAME                                  Run all fixtures for preset
  --fixture NAME                                 Run specific fixture
  --rounds N                                     Rounds (default: 2)
  --backend NAME                                 Backend (default: openai)
  --seed N                                       Seed (default: 42)
  --shuffle                                      Randomize seed per fixture
  --offline                                      Use mock backend (no API calls)
  --save                                         Save results to evals/results/
```

**Examples:**
```bash
# Test markets preset
python -m evals.run_agent_eval --preset markets --rounds 2 --save

# Offline testing
python -m evals.run_agent_eval --fixture markets_1 --offline --save

# Reproducible test
python -m evals.run_agent_eval --preset markets --seed 42 --save
```

## Features & Gotchas

### Image Handling

- **Webpage URLs**: Automatically screenshot → upload to ImageKit → use as image URL
- **Local files**: Upload to ImageKit automatically
- **Direct image URLs**: Passed through directly
- **Vision support**: Backends checked for vision capability (skips if unsupported)

### Determinism

- **Fixed seeds**: Default seed 42 for reproducibility
- **ISO 8601 timestamps**: All timestamps in UTC with Z suffix for easy diffing
- **Partial results**: Marked with `"status": "partial"` when budget exhausted
- **Token tracking**: Missing token usage returns `null` (graceful degradation)

### Error Handling

- **Settings validation**: Fails fast on schema errors (exits non-zero)
- **Vision support**: Fixtures skipped (not failed) if backend doesn't support images
- **Budget exhaustion**: Result status set to `"partial"`, not `"error"`
- **Missing tokens**: Returns `null` instead of failing

### Sentiment Analysis

Simple keyword-based heuristics (visual triage only, not production-ready):
- **Positive**: agree, support, bullish, improve, favorable, opportunity
- **Negative**: risk, caution, concern, uncertain, bearish, invalid, warning
- **Neutral**: Everything else

Color coding in dashboards: 🟢 green (positive), 🔴 red (negative), ⚪ gray (neutral)

## Troubleshooting

**"Settings validation failed"**
- Check `settings.schema.json` exists
- Install `jsonschema`: `pip install jsonschema`
- Verify all required fields present
- Remove any unknown keys

**"Backend doesn't support vision"**
- Check backend name (must be vision-capable: openai, anthropic, google)
- Use `--offline` for testing without vision

**"Budget exhausted" / "Partial results"**
- Increase `global_max_tokens` in orchestrator
- Reduce `rounds` or `max_tokens` per agent
- This is expected behavior (not an error)

**"No fixtures found"**
- Ensure fixtures in `evals/fixtures/` directory
- Use `--fixture markets_1` for specific fixture
- Or create your own fixtures following the JSON structure

## Requirements Summary

**Core:**
- Python 3.10+
- `openai`, `python-dotenv`, `selenium`, `webdriver-manager`, `Pillow`, `imagekitio`

**Dashboards:**
- `flask` (for `viz.py`)
- `streamlit` (for `viz_streamlit.py`)

**Validation:**
- `jsonschema` (for settings validation)

## License & Contributing

See parent directory for license information.

---

**Quick Reference:**
- Run: `python discuss_v12.py --preset markets`
- View: `python viz.py` → http://127.0.0.1:5000
- Test: `python -m evals.run_agent_eval --preset markets --offline --save`
