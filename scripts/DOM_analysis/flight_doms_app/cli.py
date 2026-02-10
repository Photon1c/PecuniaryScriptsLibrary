"""CLI entrypoint for the flight_doms_app wrapper.

This module provides a small, explicit command-line interface for starting
the existing Live DOM Dashboard v3 + intraday flight classifier as a Flask
application.

It does **not** implement scraping or classification itself; instead, it
relies on the adapter utilities in :mod:`flight_doms_app.adapter_dom`, which
in turn import the canonical implementation under the knowledgebase:

    ~/.openclaw/workspace/memory/knowledgebase/stock-monitor/flight_doms/

Typical usage (Linux / VPS)
---------------------------

1. Activate the `lobsterenv` virtual environment, which should include:

   - flask
   - selenium
   - webdriver-manager
   - scipy

2. Start the app from the workspace root:

   .. code-block:: bash

       cd ~/.openclaw/workspace
       python -m flight_doms_app.cli --ticker SPY --port 8021 --host 127.0.0.1

3. Open the dashboard in a browser:

   - http://127.0.0.1:8021

Windows usage
-------------

On Windows, the pattern is the same:

1. Create and activate a Python environment with the dependencies above.
2. Ensure the `flight_doms_app` package and the `flight_doms` knowledgebase
   code are on `PYTHONPATH` or installed as a package.
3. Run::

       python -m flight_doms_app.cli --ticker SPY --port 8021 --host 127.0.0.1

This keeps the run command symmetric across platforms.
"""

from __future__ import annotations

import argparse
import logging
from typing import NoReturn

from .adapter_dom import create_app_via_adapter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="flight_doms_app",
        description=(
            "Thin CLI for the Live DOM Dashboard v3 + intraday flight "
            "classifier Flask app."
        ),
    )
    parser.add_argument(
        "--ticker",
        default="SPY",
        help="Default ticker symbol for the dashboard (default: SPY)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind the Flask app (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8021,
        help="Port to bind the Flask app (default: 8021)",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help=(
            "Disable headless mode for the Selenium driver; useful for "
            "debugging on a desktop environment."
        ),
    )
    return parser.parse_args()


def main() -> NoReturn:
    """Run the Flask app using the existing DOM analyzer implementation.

    This function is the canonical entrypoint for ``python -m
    flight_doms_app.cli``.
    """

    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )

    headless = not args.no_headless
    logging.info(
        "Starting Live DOM Dashboard v3 (ticker=%s, host=%s, port=%d, headless=%s)",
        args.ticker,
        args.host,
        args.port,
        headless,
    )

    app = create_app_via_adapter(
        default_ticker=args.ticker,
        poll_interval_seconds=1,
        headless=headless,
    )

    # Note: debug=False to avoid auto-reloader complications on servers.
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":  # pragma: no cover - manual CLI invocation
    main()
