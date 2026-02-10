"""Smoke tests for DOM scraping via the `flight_doms` app adapter.

These tests answer a pragmatic question:

    "Given the current environment and dependencies, can we successfully
    invoke the DOM scraper and obtain a structurally valid snapshot?"

They deliberately avoid asserting on exact prices or DOM shapes, because the
CBOE Book Viewer structure and market conditions can change.

Behavior
--------
- If :mod:`dom_analyzer` cannot be imported, the tests will be **skipped**
  or will assert that we see a structured error.
- If Selenium / Chrome cannot be started, the tests will **not** crash the
  suite; instead, they will either be skipped or will assert that an error
  is returned in a structured way.

Run locally (Linux / lobsterenv)
--------------------------------

.. code-block:: bash

    cd ~/.openclaw/workspace/memory/knowledgebase/stock-monitor/flight_doms
    # Ensure lobsterenv is activated and dependencies are installed:
    #   pip install flask selenium webdriver-manager scipy pytest
    pytest -q app/tests/test_dom_scrape_smoke.py

On Windows, the same pattern applies once a compatible environment is set
up and this directory is importable.
"""

from __future__ import annotations

import os
import pytest

from app.adapter_dom import (
    DepthSnapshotResult,
    get_depth_snapshot_safe,
)


@pytest.mark.skipif(
    "CI" in os.environ,
    reason="DOM smoke tests rely on external browser + CBOE; skip in CI by default.",
)
def test_depth_snapshot_shape_or_structured_error() -> None:
    """Verify that the adapter returns either a valid snapshot or a clear error.

    This test does not enforce that the DOM scrape must succeed; instead, it
    asserts that we get **some** well-formed response:

    - If ``ok`` is True, ``data`` should look like a DOM snapshot.
    - If ``ok`` is False, ``error`` should contain a short diagnostic string.
    """

    result: DepthSnapshotResult = get_depth_snapshot_safe(ticker="SPY", max_levels=5)

    # Case 1: import or runtime failure -> structured error
    if not result.ok:
        assert result.data is None
        assert isinstance(result.error, str) and result.error.strip(), (
            "Expected a short error message when ok=False",
        )
        return

    # Case 2: snapshot succeeded -> basic shape checks
    data = result.data
    assert isinstance(data, dict), "Expected DOM snapshot data to be a dict when ok=True"

    # Required keys, even if values are None/empty
    for key in ("timestamp", "symbol", "last_price", "bids", "asks"):
        assert key in data, f"Missing key in DOM snapshot: {key}"

    # bids/asks should be lists
    assert isinstance(data["bids"], list)
    assert isinstance(data["asks"], list)

    # If present, individual levels should be 2-element sequences
    for side in ("bids", "asks"):
        levels = data[side]
        for level in levels:
            assert isinstance(level, (list, tuple)), f"{side} level must be list/tuple: {level!r}"
            assert len(level) == 2, f"{side} level must have length 2: {level!r}"
