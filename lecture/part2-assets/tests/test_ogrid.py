"""Level-2 O-grid recipe through the sandbox, mirroring test_agent patterns."""

from __future__ import annotations

import json
from pathlib import Path

from quadagent import quality
from quadagent.sandbox import run_python

BOARD = {
    "board_id": "ogrid-test",
    "outline": {"width": 12.0, "height": 12.0, "chamfers": []},
    "holes": [{"cx": 4.0, "cy": 4.0, "r": 1.2}, {"cx": 8.0, "cy": 8.0, "r": 1.2}],
    "cutouts": [{"cx": 8.0, "cy": 3.0, "w": 3.0, "h": 2.0}],
    "slots": [{"cx": 4.0, "cy": 8.0, "w": 1.0, "h": 2.0}],
    "header": {"cx": 6.0, "cy": 11.0, "w": 5.0, "h": 1.0},
}


def _ogrid_script_source() -> str:
    from quadagent.recipes import recipe_source

    return recipe_source("level2_ogrid")


def test_ogrid_mesh_two_holes(tmp_path: Path) -> None:
    (tmp_path / "board.json").write_text(json.dumps(BOARD))
    result, _script = run_python(_ogrid_script_source(), cwd=tmp_path, timeout_s=120, mode="auto")

    assert result.exit_code == 0, result.stderr[-500:]
    assert (tmp_path / "mesh.vtk").is_file()
    assert (tmp_path / "mesh.png").is_file()

    report = quality.analyze(
        tmp_path / "mesh.msh", expected_area_mm2=quality.analytic_area(BOARD)
    )
    assert report["quad_fraction"] >= 0.95
    assert report["scaled_jacobian_min"] > 0.4
    assert report["n_quads"] > 100
