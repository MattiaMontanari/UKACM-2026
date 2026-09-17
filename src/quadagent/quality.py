"""Small, trusted quality report for generated Gmsh meshes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

TRIANGLE = 2
QUAD = 3


def _load_mesh(path: Path) -> tuple[dict[int, np.ndarray], np.ndarray, int]:
    """Load 2D element connectivity and node coordinates from a Gmsh mesh."""
    import gmsh

    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        points = np.asarray(coordinates, dtype=float).reshape(-1, 3)[:, :2]
        rows = {int(tag): index for index, tag in enumerate(node_tags)}
        parts: dict[int, list[np.ndarray]] = {}
        for _dimension, entity in gmsh.model.getEntities(2):
            element_types, _, element_nodes = gmsh.model.mesh.getElements(2, entity)
            for element_type, connectivity in zip(
                element_types, element_nodes, strict=True
            ):
                size = gmsh.model.mesh.getElementProperties(int(element_type))[3]
                block = np.fromiter(
                    (rows[int(tag)] for tag in connectivity), dtype=np.int64
                ).reshape(-1, size)
                parts.setdefault(int(element_type), []).append(block)
        return (
            {kind: np.vstack(blocks) for kind, blocks in parts.items()},
            points,
            len(node_tags),
        )
    finally:
        gmsh.clear()
        gmsh.finalize()


def _scaled_jacobian(quads: np.ndarray) -> np.ndarray:
    """Return the minimum corner scaled Jacobian for each quadrilateral."""
    edges = np.roll(quads, -1, axis=1) - quads
    previous = np.roll(edges, 1, axis=1)
    cross = previous[:, :, 0] * edges[:, :, 1] - previous[:, :, 1] * edges[:, :, 0]
    denominator = np.linalg.norm(previous, axis=2) * np.linalg.norm(edges, axis=2)
    return np.min(cross / np.maximum(denominator, 1e-300), axis=1)


def analytic_area(board: dict[str, Any]) -> float:
    """Return board outline area minus its holes and rectangular voids."""
    outline = board["outline"]
    area = float(outline["width"]) * float(outline["height"])
    area -= sum(
        float(chamfer["size"]) ** 2 / 2 for chamfer in outline.get("chamfers", [])
    )
    area -= sum(
        math.pi * float(hole["r"]) ** 2 for hole in board.get("holes", [])
    )
    for name in ("cutouts", "slots"):
        area -= sum(
            float(void["w"]) * float(void["h"]) for void in board.get(name, [])
        )
    if header := board.get("header"):
        area -= float(header["w"]) * float(header["h"])
    return area


def analyze(mesh_path: str | Path, *, expected_area_mm2: float) -> dict[str, Any]:
    """Return the compact quality report used by the agent."""
    path = Path(mesh_path)
    if not path.is_file():
        raise FileNotFoundError(f"mesh file not found: {path}")
    blocks, points, node_count = _load_mesh(path)
    if not blocks:
        raise ValueError(f"no 2D elements found in {path}")

    quad_rows = blocks.get(QUAD, np.empty((0, 4), dtype=np.int64))
    triangle_rows = blocks.get(TRIANGLE, np.empty((0, 3), dtype=np.int64))
    element_count = sum(len(block) for block in blocks.values())
    mesh_area = 0.0
    for block in blocks.values():
        polygon = points[block]
        x = polygon[:, :, 0]
        y = polygon[:, :, 1]
        signed_area = 0.5 * np.sum(
            x * (np.roll(y, -1, axis=1) - np.roll(y, 1, axis=1)), axis=1
        )
        mesh_area += float(np.abs(signed_area).sum())

    return {
        "mesh_file": path.name,
        "n_nodes": node_count,
        "n_elements_2d": element_count,
        "n_quads": len(quad_rows),
        "n_triangles": len(triangle_rows),
        "quad_fraction": round(len(quad_rows) / element_count, 4),
        "scaled_jacobian_min": (
            round(float(_scaled_jacobian(points[quad_rows]).min()), 4)
            if len(quad_rows)
            else None
        ),
        "mesh_area_mm2": round(mesh_area, 3),
        "expected_area_mm2": round(expected_area_mm2, 3),
        "area_ratio": round(mesh_area / expected_area_mm2, 4),
    }


def summarize(report: dict[str, Any]) -> str:
    """Format a compact report for the model and terminal."""
    return "\n".join(f"{name}: {value}" for name, value in report.items())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh_file", type=Path)
    parser.add_argument("--expected-area", type=float, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.mesh_file, expected_area_mm2=args.expected_area)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
