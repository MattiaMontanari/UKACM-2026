"""Contract tests for the real-GLM milestone validator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.live_glm_test import ValidationError, validate_workspace


def test_live_runner_is_executable_as_a_script() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/live_glm_test.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _write_valid_workspace(path: Path) -> None:
    for name in ("mesh.py", "mesh.msh", "mesh.vtk", "mesh.png"):
        (path / name).write_bytes(b"artifact")
    (path / "todos.json").write_text("[]\n")
    (path / "transcript.json").write_text("[]\n")
    (path / "quality.json").write_text(
        json.dumps(
            {
                "n_elements_2d": 100,
                "n_quads": 95,
                "scaled_jacobian_min": 0.2,
                "area_ratio": 1.001,
            }
        )
    )
    (path / "run.json").write_text(
        json.dumps(
            {
                "model_calls": 6,
                "tool_calls": [
                    "write_todos",
                    "read_file",
                    "write_file",
                    "execute",
                    "mesh_quality",
                ],
                "usage": {"total_tokens": 1234},
                "error": None,
            }
        )
    )


def test_validate_workspace_accepts_the_live_contract(tmp_path: Path) -> None:
    _write_valid_workspace(tmp_path)
    result = validate_workspace(tmp_path)
    assert result["quality"]["n_quads"] == 95
    assert result["run"]["usage"]["total_tokens"] == 1234


def test_validate_workspace_rejects_nonterminal_scoring(tmp_path: Path) -> None:
    _write_valid_workspace(tmp_path)
    run = json.loads((tmp_path / "run.json").read_text())
    run["tool_calls"].append("execute")
    (tmp_path / "run.json").write_text(json.dumps(run))

    with pytest.raises(ValidationError, match="terminal tool call"):
        validate_workspace(tmp_path)
