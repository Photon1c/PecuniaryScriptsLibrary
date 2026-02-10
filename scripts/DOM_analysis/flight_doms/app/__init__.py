"""app package for flight_doms.

Thin, testable wrapper around the existing DOM analyzer + intraday flight
classifier implementation in this directory.

Goals
-----
- Avoid re-implementing scraping or classifier logic.
- Provide a small, well-documented CLI entrypoint that works on Linux first
  (inside the `lobsterenv` virtual environment), and later on Windows.
- Offer smoke tests to validate DOM retrieval and error handling without
  adding fragility to the system.

Directory layout
----------------
This package lives alongside the canonical implementation:

    flight_doms/
      dom_analyzer.py
      intraday_flight_phase.py
      app/
        __init__.py
        adapter_dom.py
        cli.py
        tests/

Typical usage (Linux / VPS)
---------------------------

.. code-block:: bash

   # 1. Activate lobsterenv (with flask, selenium, webdriver-manager, scipy)
   source ~/.openclaw/workspace/lobsterenv/bin/activate

   # 2. Launch the app from the flight_doms directory
   cd ~/.openclaw/workspace/memory/knowledgebase/stock-monitor/flight_doms
   python -m app.cli --ticker SPY --port 8021 --host 127.0.0.1

On Windows, the pattern is identical once a suitable Python environment is
created and the `flight_doms` directory is on `PYTHONPATH`.
"""

__all__ = [
    "__version__",
]

__version__ = "0.1.0"  # local-only semantic version for this wrapper
