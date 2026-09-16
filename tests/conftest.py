"""Shared test configuration: warm matplotlib cache + isolation asserts."""

from __future__ import annotations

import os
from pathlib import Path

_cache = Path(__file__).parent / "_mplcache"
_cache.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache.resolve()))
