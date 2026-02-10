"""DOM adapter utilities for the `flight_doms` app wrapper.

This module provides a thin, testable interface around the existing DOM
scraper and intraday flight classifier code in this directory.

It:
- Avoids duplicating scraping logic.
- Exposes a small API for the CLI and tests to call.
- Keeps failure modes explicit and fail-soft.

Environment expectations
------------------------
- Linux / VPS: run inside the `lobsterenv` virtual environment with:

  - flask
  - selenium
  - webdriver-manager
  - scipy

- Windows: create a comparable environment and ensure this `flight_doms`
  directory is importable (e.g., on `PYTHONPATH`).

The scraping and classifier logic remain in :mod:`dom_analyzer` and
:mod:`intraday_flight_phase`; this adapter just wires them up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from flask import Flask

# Local imports from the canonical implementation in this directory.
try:
    from .. import dom_analyzer
except Exception as exc:  # pragma: no cover - import failure handled via flags
    _IMPORT_ERROR = exc
    dom_analyzer = None  # type: ignore[assignment]
else:
    _IMPORT_ERROR = None


@dataclass
class DepthSnapshotResult:
    """Wrapper type for a DOM depth snapshot call via the adapter.

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
    """

    if _IMPORT_ERROR is not None or dom_analyzer is None:
        return DepthSnapshotResult(
            ok=False,
            data=None,
            error=f"dom_analyzer import failed: {_IMPORT_ERROR}",
        )

    try:
        scraper = dom_analyzer.get_global_scraper(headless=True)
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

    This function simply forwards to :func:`dom_analyzer.create_app`. It
    exists so callers in this `app` package do not need to know where the
    underlying implementation lives.

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
        If the underlying implementation is not importable.
    """

    if _IMPORT_ERROR is not None or dom_analyzer is None:
        raise RuntimeError(f"dom_analyzer.create_app not available: {_IMPORT_ERROR}")

    app: Flask = dom_analyzer.create_app(
        default_ticker=default_ticker,
        poll_interval_seconds=poll_interval_seconds,
        headless=headless,
        replayer=None,
    )
    return app
