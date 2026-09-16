"""Level-3 pure block-structured mesh through the sandbox.

Fixture board: two holes, one cutout, one slot, a header and chamfers on
two corners (sw + ne). Both hole boxes are sliced by foreign void-edge
cuts (exercising the multi-curve transfinite chains), the sw corner is
absorbed northward, the ne corner southward. A second, smaller board
forces the EAST absorption branch by parking the header directly above
the sw chamfer corner square.

Note on the min-angle bound: the 90-degree angle at every hole-box
corner splits between the two adjacent quadrant patches, so the radial
connector construction caps the achievable minimum interior angle at
45 degrees (measured ~44 on the symmetric fixture). The assertion uses
40 degrees accordingly.
"""

from __future__ import annotations

import json
from pathlib import Path

from quadagent import quality
from quadagent.sandbox import run_python

BOARD = {
    "board_id": "block-test",
    "outline": {
        "width": 36.0,
        "height": 28.0,
        "chamfers": [
            {"corner": "sw", "size": 4.0},
            {"corner": "ne", "size": 3.0},
        ],
    },
    "holes": [
        {"cx": 8.0, "cy": 8.0, "r": 1.5},
        {"cx": 28.0, "cy": 20.0, "r": 1.5},
    ],
    "cutouts": [{"cx": 18.0, "cy": 8.0, "w": 4.0, "h": 3.0}],
    "slots": [{"cx": 18.0, "cy": 17.5, "w": 1.6, "h": 5.0}],
    "header": {"cx": 18.0, "cy": 23.8, "w": 8.0, "h": 1.2},
}

BOARD_EAST_ABSORB = {
    "board_id": "block-east-test",
    "outline": {"width": 30.0, "height": 30.0, "chamfers": [{"corner": "sw", "size": 2.5}]},
    "holes": [{"cx": 10.0, "cy": 10.0, "r": 1.5}],
    "cutouts": [],
    "slots": [],
    "header": {"cx": 2.5, "cy": 3.5, "w": 5.0, "h": 2.0},
}


def _block_script_source() -> str:
    from quadagent.recipes import recipe_source

    return recipe_source("level3_block")


def test_blockstructured_pure_quads(tmp_path: Path) -> None:
    (tmp_path / "board.json").write_text(json.dumps(BOARD))
    result, _script = run_python(_block_script_source(), cwd=tmp_path, timeout_s=180, mode="auto")

    assert result.exit_code == 0, result.stderr[-800:]
    assert (tmp_path / "mesh.vtk").is_file()
    assert (tmp_path / "mesh.png").is_file()
    assert (tmp_path / "mesh.msh").is_file()

    report = quality.analyze(
        tmp_path / "mesh.msh", expected_area_mm2=quality.analytic_area(BOARD)
    )
    assert report["quad_fraction"] == 1.0
    assert report["n_triangles"] == 0
    assert report["scaled_jacobian_min"] > 0.5
    assert report["min_angle_deg"] > 40.0
    assert report["area_ratio"] == 1.0 or abs(report["area_ratio"] - 1.0) < 0.005


def test_blockstructured_east_absorption(tmp_path: Path) -> None:
    (tmp_path / "board.json").write_text(json.dumps(BOARD_EAST_ABSORB))
    result, _script = run_python(_block_script_source(), cwd=tmp_path, timeout_s=180, mode="auto")

    assert result.exit_code == 0, result.stderr[-800:]
    report = quality.analyze(
        tmp_path / "mesh.msh", expected_area_mm2=quality.analytic_area(BOARD_EAST_ABSORB)
    )
    assert report["quad_fraction"] == 1.0
    assert report["n_triangles"] == 0
    assert report["scaled_jacobian_min"] > 0.5
