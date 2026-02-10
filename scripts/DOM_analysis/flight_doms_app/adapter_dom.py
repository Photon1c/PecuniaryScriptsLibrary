"""DOM adapter utilities for flight_doms_app.

This module provides a thin, testable interface around the existing DOM
scraper and intraday flight classifier code that lives under the
knowledgebase:

    ~/.openclaw/workspace/memory/knowledgebase/stock-monitor/flight_doms/

The goal is to:
- Avoid duplicating scraping logic.
- Provide a small, well-defined API that the CLI and tests can call.
- Keep failure modes explicit and fail-soft (returning structured errors
  rather than raising unhandled exceptions where possible).

Intended usage patterns
-----------------------
- Linux / VPS:
    - Activate the `lobsterenv` virtual environment.
    - Use `get_depth_snapshot_safe()` in smoke tests.
    - Use `create_app_via_adapter()` in the CLI to start the Flask app.

- Windows:
    - Create a compatible Python environment.
    - Reuse the same adapter functions via `flight_doms_app.cli`.

This file deliberately contains no network/DOM-specific constants; those
remain in the original `dom_analyzer.py` so there is a single source of
truth about how the CBOE Book Viewer is scraped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from flask import Flask

# NOTE:
# Users often start the CLI from the workspace root (or other directories),
# which means the sibling `flight_doms/` package may not be on `sys.path`.
# We therefore attempt a normal import first, then fall back to a local
# path-based import rooted at this file's directory.
import sys
from pathlib import Path

# Import from the existing knowledgebase code.
#
# The knowledgebase tree is rooted under the **workspace** directory, so the
# correct import path starts with `memory.knowledgebase` when this package is
# imported from within `~/.openclaw/workspace`.
try:
    from flight_doms import dom_analyzer
    get_global_scraper = dom_analyzer.get_global_scraper
    create_app = dom_analyzer.create_app
except Exception as exc:  # pragma: no cover - import failure is handled via flags
    try:
        here = Path(__file__).resolve()
        suite_dir = here.parent  # .../flight_doms_suite/flight_doms_app
        # Add `.../flight_doms_suite` so `import flight_doms` works.
        sys.path.insert(0, str(suite_dir.parent))
        from flight_doms import dom_analyzer  # type: ignore

        get_global_scraper = dom_analyzer.get_global_scraper  # type: ignore[attr-defined]
        create_app = dom_analyzer.create_app  # type: ignore[attr-defined]
    except Exception as exc2:
        get_global_scraper = None  # type: ignore[assignment]
        create_app = None  # type: ignore[assignment]
        _IMPORT_ERROR = exc2
    else:
        _IMPORT_ERROR = None
else:
    _IMPORT_ERROR = None


@dataclass
class DepthSnapshotResult:
    """Wrapper type for a DOM depth snapshot call.

    Attributes
    ----------
    ok:
        Whether the call succeeded.
    data:
        The raw dict returned by the scraper (timestamp, symbol, bids, asks,
        etc.), or ``None`` if the call failed before scraping.
    error:
        A short, human-readable error message if ``ok`` is False.
    """

    ok: bool
    data: Optional[Dict[str, Any]]
    error: Optional[str] = None


def get_depth_snapshot_safe(ticker: str = "SPY", max_levels: int = 5) -> DepthSnapshotResult:
    """Attempt to retrieve a single DOM depth snapshot in a fail-soft way.

    This function wraps the existing global scraper and returns a
    :class:`DepthSnapshotResult` instead of raising exceptions. It is
    intended for use in smoke tests and diagnostics.

    Parameters
    ----------
    ticker:
        Symbol to request from the CBOE Book Viewer. Defaults to ``"SPY"``.
    max_levels:
        Maximum number of book levels (bid/ask) to request.

    Notes
    -----
    - If the underlying knowledgebase modules cannot be imported, this will
      return ``ok=False`` with an explanatory error.
    - If Selenium/Chrome cannot be started or the page structure is
      incompatible, this will also return ``ok=False``.
    """

    if _IMPORT_ERROR is not None or get_global_scraper is None:
        return DepthSnapshotResult(
            ok=False,
            data=None,
            error=f"dom_analyzer import failed: {_IMPORT_ERROR}",
        )

    try:
        scraper = get_global_scraper(headless=True)
        scraper.ensure_ticker(ticker)
        data = scraper.get_depth_snapshot(max_levels=max_levels)
        return DepthSnapshotResult(ok=True, data=data, error=None)
    except Exception as exc:  # pragma: no cover - depends on external browser
        return DepthSnapshotResult(
            ok=False,
            data=None,
            error=f"depth snapshot error: {exc}",
        )


def create_app_via_adapter(
    default_ticker: str = "SPY",
    poll_interval_seconds: int = 1,
    headless: bool = True,
) -> Flask:
    """Create the Flask app using the existing DOM analyzer factory.

    This function simply forwards to :func:`create_app` from the original
    ``dom_analyzer.py`` module. It exists so callers in this package do not
    need to know where the underlying implementation lives.

    Parameters
    ----------
    default_ticker:
        Default symbol for the dashboard.
    poll_interval_seconds:
        Poll interval to pass through; currently unused in the underlying
        implementation but kept for future compatibility.
    headless:
        Whether to initialize the global scraper in headless mode when
        needed.

    Raises
    ------
    RuntimeError
        If the underlying ``create_app`` implementation is not importable.
    """

    if _IMPORT_ERROR is not None or create_app is None:
        raise RuntimeError(f"dom_analyzer.create_app not available: {_IMPORT_ERROR}")

    # The underlying create_app already handles scraper construction and
    # request routing. We simply return the Flask app instance.
    app: Flask = create_app(
        default_ticker=default_ticker,
        poll_interval_seconds=poll_interval_seconds,
        headless=headless,
        replayer=None,
    )
    return app
