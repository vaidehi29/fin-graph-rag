"""
COMPONENT : Evaluation / tests - pytest configuration
PURPOSE   : Keep the default test run offline: tests marked `integration`
            (live Neo4j, Gemini, local model) are skipped unless they are
            selected explicitly with `-m integration` (or FGR_RUN_INTEGRATION=1).
INPUTS    : pytest command line / environment
OUTPUTS   : skip markers on integration tests in offline runs
DEPENDS ON: pytest
PANEL NOTE: pipeline.txt runs unit tests with --ignore-glob="*integration*"
            and integration tests with -m integration; this also makes a bare
            `pytest` safe (no database writes, no API quota spent).
"""

import os

import pytest


def pytest_collection_modifyitems(config, items):
    selected = "integration" in (config.getoption("markexpr") or "")
    if selected or os.getenv("FGR_RUN_INTEGRATION") == "1":
        return
    skip = pytest.mark.skip(reason="live services; run with: pytest -m integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
