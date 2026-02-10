"""flight_doms_app

Thin, testable wrapper around the existing DOM analyzer + intraday flight
classifier code that lives under the knowledgebase.

This package is intentionally small and modular:

- It does **not** re-implement scraping or flight logic.
- It imports and adapts the existing `dom_analyzer.py` and
  `intraday_flight_phase.py` code under:

    ~/.openclaw/workspace/memory/knowledgebase/stock-monitor/flight_doms/

- It exposes a simple CLI entrypoint that starts the Flask app, suitable for
  Linux (first) and later Windows usage.
- It provides a small `tests/` tree to smoke-test DOM retrieval and basic
  classifier integration.

Environment expectations
------------------------
- For Linux, run inside the `lobsterenv` virtual environment, which should
  contain the heavy dependencies (selenium, webdriver-manager, flask, scipy).
- For Windows, create a Python environment with the same dependencies and
  run `python -m flight_doms_app.cli`.

This module intentionally avoids hard-coding OS-specific paths; callers are
responsible for activating the correct environment before running it.
"""

__all__ = [
    "__version__",
]

__version__ = "0.1.0"  # local-only semantic version for this wrapper
