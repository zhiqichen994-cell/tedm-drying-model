"""Shared pytest fixtures and configuration for the TEDM test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# make the package importable when running ``pytest`` from the repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: full-model runs (tens of seconds); deselect with -m 'not slow'"
    )
