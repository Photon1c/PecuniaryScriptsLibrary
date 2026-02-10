"""Tests for :mod:`flight_doms_app`.

The test suite is intentionally small and focused on **smoke testing** that
matters for operational reliability rather than exhaustive unit coverage.

Key goals
---------
- Verify that the DOM scraping stack can be invoked from this wrapper
  (subject to Selenium / browser availability).
- Verify that failure modes are captured as structured results instead of
  unhandled exceptions.

These tests are written with Linux/VPS usage in mind first. On Windows, the
same tests can be reused once an appropriate environment (Python, Selenium,
Chrome/Edge) is available.
"""
