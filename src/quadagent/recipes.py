"""Host-side access to the maintainer-certified recipes in `recipes/`.

These scripts are deliberately OUTSIDE the installed `quadagent` package so
the sandboxed agent cannot import them: they are the reference solutions
for sweeps and tests, never agent-visible code. The agent gets the
single-hole PRIMITIVES as knowledge (references/playbook.md) and must
compose its own whole-domain strategy.
"""

from __future__ import annotations

from pathlib import Path

RECIPE_NAMES = ("level0_baseline", "level2_ogrid", "level3_block")

RECIPE_DIR = Path(__file__).resolve().parents[2] / "recipes"


def recipe_path(name: str) -> Path:
    """Absolute path of one certified recipe script."""
    if name not in RECIPE_NAMES:
        raise KeyError(f"unknown recipe {name!r}; available: {RECIPE_NAMES}")
    path = RECIPE_DIR / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"recipe not found: {path}")
    return path


def recipe_source(name: str) -> str:
    """Source text of one certified recipe script."""
    return recipe_path(name).read_text()
