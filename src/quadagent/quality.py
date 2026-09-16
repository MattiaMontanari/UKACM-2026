"""Host-side (trusted) mesh quality metrics for 2D explicit-FEA meshes.

Reads a Gmsh-producible mesh file (.msh or .vtk), keeps only dim-2
elements, and computes the quad metrics that matter for explicit
dynamics: quad fraction, scaled Jacobian, aspect ratio, interior angles,
edge lengths, and the critical time-step proxy dt ~ h_min / c.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

TRI_TYPE = 2
QUAD_TYPE = 3
DEFAULT_WAVE_SPEED_M_S = 3000.0

GATES = {
    "quad_fraction": (0.98, 0.90, 0.80),
    "scaled_jacobian_min": (0.60, 0.40, 0.0),
    "aspect_ratio_max": (3.0, 5.0, 10.0),
    "min_angle_deg": (45.0, 35.0, 20.0),
    "element_count": (10_000, 20_000, 50_000),
}


def _load_elements(
    path: Path,
) -> tuple[dict[int, np.ndarray], int, np.ndarray, dict[int, dict[int, np.ndarray]]]:
    """Return global {etype: rows}, node count, coords, and per-surface blocks."""
    import gmsh

    # Deep Agents invokes tools in a worker thread. Disabling Gmsh's SIGINT
    # handler keeps initialization valid outside Python's main thread.
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coords, dtype=float).reshape(-1, 3)[:, :2]
        tag_to_row = {int(t): i for i, t in enumerate(node_tags)}
        global_parts: dict[int, list[np.ndarray]] = {}
        entity_blocks: dict[int, dict[int, np.ndarray]] = {}
        for _dim, tag in gmsh.model.getEntities(2):
            etypes, _, enodes = gmsh.model.mesh.getElements(2, tag)
            small: dict[int, np.ndarray] = {}
            for etype, conn in zip(etypes, enodes, strict=True):
                nodes_per_elem = gmsh.model.mesh.getElementProperties(int(etype))[3]
                rows = (
                    np.array([tag_to_row[int(t)] for t in conn], dtype=np.int64)
                    .reshape(-1, nodes_per_elem)
                )
                small[int(etype)] = rows
                global_parts.setdefault(int(etype), []).append(rows)
            if small:
                entity_blocks[int(tag)] = small
        blocks = {t: np.vstack(v) for t, v in global_parts.items()}
        return blocks, len(node_tags), coords, entity_blocks
    finally:
        gmsh.clear()
        gmsh.finalize()


def _boundary_parity(rows_by_type: dict[int, np.ndarray]) -> tuple[int, bool]:
    """Boundary node count of one surface's element patch and its parity."""
    edge_count: dict[tuple[int, int], int] = {}
    for rows in rows_by_type.values():
        for elem in rows:
            for a, b in zip(elem, np.roll(elem, -1), strict=False):
                key = (int(min(a, b)), int(max(a, b)))
                edge_count[key] = edge_count.get(key, 0) + 1
    nodes: set[int] = set()
    for (a, b), count in edge_count.items():
        if count == 1:
            nodes.update((a, b))
    return len(nodes), len(nodes) % 2 == 1


def _quad_metrics(q: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized quad metrics for corners (n, 4, 2) in CCW order."""
    edges = np.roll(q, -1, axis=1) - q  # (n, 4, 2): edge k starts at corner k
    lengths = np.linalg.norm(edges, axis=2)
    cross = edges[:, 3, 0] * edges[:, 0, 1] - edges[:, 3, 1] * edges[:, 0, 0]
    for k in range(3):
        cross = np.minimum(
            cross,
            edges[:, k, 0] * edges[:, k + 1, 1] - edges[:, k, 1] * edges[:, k + 1, 0],
        )
    denom = lengths[:, 3] * lengths[:, 0]
    for k in range(3):
        denom = np.minimum(denom, lengths[:, k] * lengths[:, k + 1])
    scaled_jacobian = cross / np.maximum(denom, 1e-300)

    incoming = -np.roll(edges, 1, axis=1)  # vector from corner k back to k-1
    dot = np.sum(incoming * edges, axis=2)
    norm = np.linalg.norm(incoming, axis=2) * np.linalg.norm(edges, axis=2)
    cos = np.clip(dot / np.maximum(norm, 1e-300), -1.0, 1.0)
    angles = np.degrees(np.arccos(cos))

    aspect = lengths.max(axis=1) / np.maximum(lengths.min(axis=1), 1e-300)
    return {
        "scaled_jacobian": scaled_jacobian,
        "min_angle_deg": angles.min(axis=1),
        "aspect_ratio": aspect,
        "min_edge": lengths.min(axis=1),
        "mean_edge": lengths.mean(axis=1),
    }


def analytic_area(payload: dict[str, Any]) -> float:
    """Analytic domain area of a board payload (outline minus every void)."""
    outline = payload["outline"]
    area = float(outline["width"]) * float(outline["height"])
    area -= sum(c["size"] ** 2 / 2.0 for c in outline.get("chamfers", []))
    area -= sum(math.pi * h["r"] ** 2 for h in payload.get("holes", []))
    for key in ("cutouts", "slots"):
        area -= sum(r["w"] * r["h"] for r in payload.get(key, []))
    header = payload.get("header")
    if header:
        area -= float(header["w"]) * float(header["h"])
    return area


def analyze(
    mesh_path: str | Path,
    *,
    expected_area_mm2: float | None = None,
    wave_speed_m_s: float = DEFAULT_WAVE_SPEED_M_S,
) -> dict[str, Any]:
    """Compute the quality report dictionary for one mesh file."""
    path = Path(mesh_path)
    if not path.is_file():
        raise FileNotFoundError(f"mesh file not found: {path}")
    blocks, n_nodes, coords, entity_blocks = _load_elements(path)
    if not blocks:
        raise ValueError(f"no 2D elements found in {path}")

    report: dict[str, Any] = {
        "mesh_file": path.name,
        "n_nodes": n_nodes,
        "counts": {str(t): int(m.shape[0]) for t, m in blocks.items()},
    }
    n_quads = blocks.get(QUAD_TYPE, np.zeros((0, 4), dtype=np.int64)).shape[0]
    n_tris = blocks.get(TRI_TYPE, np.zeros((0, 3), dtype=np.int64)).shape[0]
    n_2d = sum(m.shape[0] for m in blocks.values())
    report["n_elements_2d"] = n_2d
    report["n_quads"] = n_quads
    report["n_triangles"] = n_tris
    report["quad_fraction"] = round(n_quads / n_2d, 4) if n_2d else 0.0

    all_edges: list[np.ndarray] = []
    if n_quads:
        quads = coords[blocks[QUAD_TYPE]]
        metrics = _quad_metrics(quads)
        report["scaled_jacobian_min"] = round(float(metrics["scaled_jacobian"].min()), 4)
        report["scaled_jacobian_mean"] = round(float(metrics["scaled_jacobian"].mean()), 4)
        report["aspect_ratio_max"] = round(float(metrics["aspect_ratio"].max()), 3)
        report["min_angle_deg"] = round(float(metrics["min_angle_deg"].min()), 2)
        report["aspect_ratio_p95"] = round(float(np.percentile(metrics["aspect_ratio"], 95)), 3)
        all_edges.append(metrics["mean_edge"])
    if n_tris:
        tris = coords[blocks[TRI_TYPE]]
        e = np.stack(
            [np.roll(tris, -1, axis=1) - tris, np.roll(tris, -2, axis=1) - tris],
            axis=2,
        )
        norms = np.linalg.norm(e, axis=3)
        cos = np.clip(
            np.sum(e[:, :, 0] * e[:, :, 1], axis=2)
            / np.maximum(norms[:, :, 0] * norms[:, :, 1], 1e-300),
            -1.0,
            1.0,
        )
        report["triangle_min_angle_deg"] = round(float(np.degrees(np.arccos(cos)).min()), 2)
        all_edges.append(norms.reshape(-1))

    if expected_area_mm2 is not None:
        area = 0.0
        for matrix in blocks.values():
            poly = coords[matrix]
            xs, ys = poly[:, :, 0], poly[:, :, 1]
            signed = 0.5 * np.sum(xs * (np.roll(ys, -1, axis=1) - np.roll(ys, 1, axis=1)), axis=1)
            area += float(np.abs(signed).sum())
        report["mesh_area_mm2"] = round(area, 3)
        report["expected_area_mm2"] = round(expected_area_mm2, 3)
        report["area_ratio"] = round(area / expected_area_mm2, 4) if expected_area_mm2 else None

    if all_edges:
        edges = np.concatenate(all_edges)
        h_min = float(edges.min())
        report["edge_len_min_mm"] = round(h_min, 4)
        report["edge_len_mean_mm"] = round(float(edges.mean()), 4)
        report["edge_len_max_mm"] = round(float(edges.max()), 4)
        dt_s = h_min * 1e-3 / wave_speed_m_s
        report["wave_speed_m_s"] = wave_speed_m_s
        report["dt_critical_us"] = round(dt_s * 1e6, 4)

    if n_quads:
        sj = _quad_metrics(coords[blocks[QUAD_TYPE]])["scaled_jacobian"]
        order = np.argsort(sj)[:3]
        report["worst_quads"] = [
            {
                "scaled_jacobian": round(float(sj[i]), 4),
                "x": round(float(coords[blocks[QUAD_TYPE]][i, :, 0].mean()), 2),
                "y": round(float(coords[blocks[QUAD_TYPE]][i, :, 1].mean()), 2),
            }
            for i in order
        ]

    surfaces = []
    for tag, small in sorted(entity_blocks.items()):
        n_q = small.get(QUAD_TYPE, np.zeros((0, 4), dtype=np.int64)).shape[0]
        n_t = small.get(TRI_TYPE, np.zeros((0, 3), dtype=np.int64)).shape[0]
        sj_min = None
        if n_q:
            sj_min = round(float(_quad_metrics(coords[small[QUAD_TYPE]])["scaled_jacobian"].min()), 4)
        boundary, odd = _boundary_parity(small)
        surfaces.append(
            {
                "surface": tag,
                "quads": n_q,
                "tris": n_t,
                "scaled_jacobian_min": sj_min,
                "boundary_nodes": boundary,
                "odd_parity": odd,
            }
        )
    report["surfaces"] = surfaces
    report["n_surfaces"] = len(surfaces)
    return report


def gate_status(metric: str, value: float, better: str) -> str:
    """Classify one metric as pass/warn/fail against its (target, acceptable) gate."""
    target, acceptable, _fail = GATES[metric]
    high = better == "high"
    meets = (lambda v: v >= target) if high else (lambda v: v <= target)
    tolerates = (lambda v: v >= acceptable) if high else (lambda v: v <= acceptable)
    if meets(value):
        return "pass"
    return "warn" if tolerates(value) else "fail"


def summarize(report: dict[str, Any]) -> str:
    """Render the report plus gate verdicts as agent-facing text."""
    lines = [f"mesh quality report: {report['mesh_file']}"]
    for key in (
        "n_nodes",
        "n_elements_2d",
        "n_quads",
        "n_triangles",
        "quad_fraction",
        "scaled_jacobian_min",
        "scaled_jacobian_mean",
        "aspect_ratio_max",
        "aspect_ratio_p95",
        "min_angle_deg",
        "triangle_min_angle_deg",
        "edge_len_min_mm",
        "edge_len_mean_mm",
        "edge_len_max_mm",
        "dt_critical_us",
        "mesh_area_mm2",
        "expected_area_mm2",
        "area_ratio",
    ):
        if key in report:
            lines.append(f"  {key}: {report[key]}")
    checks = [
        ("quad_fraction", report.get("quad_fraction", 0.0), "high"),
        (
            "scaled_jacobian_min",
            report.get("scaled_jacobian_min", -1.0),
            "high",
        ),
        ("aspect_ratio_max", report.get("aspect_ratio_max", math.inf), "low"),
        ("min_angle_deg", report.get("min_angle_deg", 0.0), "high"),
        ("element_count", report.get("n_elements_2d", math.inf), "count"),
    ]
    lines.append("gates (target / acceptable / hard-fail):")
    for name, value, better in checks:
        status = gate_status(name, float(value), better)
        mark = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[status]
        lines.append(f"  [{mark}] {name} = {value}")
    for worst in report.get("worst_quads", [])[:3]:
        lines.append(
            f"  worst quad: SJ {worst['scaled_jacobian']} at ({worst['x']}, {worst['y']})"
        )
    flagged = [
        s
        for s in report.get("surfaces", [])
        if s["odd_parity"] or s["tris"]
    ]
    lines.append(f"surfaces: {report.get('n_surfaces', 0)} total")
    for s in flagged[:6]:
        why = []
        if s["tris"]:
            why.append(f"{s['tris']} tris")
        if s["odd_parity"]:
            why.append(f"odd boundary parity ({s['boundary_nodes']} nodes) -> cannot fully recombine")
        lines.append(f"  surface {s['surface']}: {', '.join(why)}")
    if not flagged:
        lines.append("  all surfaces even parity, no triangles")
    return "\n".join(lines)


def summarize_json(report: dict[str, Any]) -> str:
    """Compact JSON rendering for the transcript."""
    return json.dumps(report, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    """Run the trusted scorer in a dedicated process and emit JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_file", type=Path)
    parser.add_argument("--expected-area", type=float, required=True)
    args = parser.parse_args(argv)
    print(summarize_json(analyze(args.mesh_file, expected_area_mm2=args.expected_area)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
