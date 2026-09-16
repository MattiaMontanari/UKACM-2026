"""Render comparison PNG panels: certified recipes (L0/L2/L3) vs agent runs.

Usage:
    uv run python scripts/compare_panels.py

Outputs workspace/comparisons/<board>_levels_vs_agent.png. Certified meshes
come from workspace/block_sweep (t=1.0); agent meshes from their session
workspaces (t=2.0). Panel titles carry the key metrics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gmsh

from quadagent import quality

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "workspace" / "comparisons"


def load_mesh_polys(mesh_path: Path) -> tuple[list[list[tuple[float, float]]], list[list[tuple[float, float]]]]:
    """Return (quads, tris) coordinate polygons of a mesh file."""
    gmsh.initialize()
    try:
        gmsh.open(str(mesh_path))
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        lookup = {int(t): (coords[3 * i], coords[3 * i + 1]) for i, t in enumerate(node_tags)}
        etypes, _, enodes = gmsh.model.mesh.getElements(2)
        quads: list[list[tuple[float, float]]] = []
        tris: list[list[tuple[float, float]]] = []
        for etype, conn in zip(etypes, enodes, strict=True):
            n = 4 if etype == 3 else 3
            for row in zip(*[iter(conn)] * n, strict=False):
                poly = [lookup[int(t)] for t in row]
                (quads if n == 4 else tris).append(poly)
        return quads, tris
    finally:
        gmsh.clear()
        gmsh.finalize()


def metrics_for(directory: Path, mesh: Path) -> dict[str, Any]:
    qfile = directory / "quality.json"
    if qfile.is_file():
        return json.loads(qfile.read_text())
    return quality.analyze(mesh)


def draw_panel(ax: plt.Axes, mesh: Path, title: str, metrics: dict[str, Any], window: tuple[float, float, float, float] | None) -> None:
    quads, tris = load_mesh_polys(mesh)
    if quads:
        ax.add_collection(PolyCollection(quads, facecolor="none", edgecolor="0.55", linewidth=0.35))
    if tris:
        ax.add_collection(PolyCollection(tris, facecolor="red", edgecolor="red", alpha=0.6, linewidth=0.35))
    ax.autoscale()
    if window:
        ax.set_xlim(window[0], window[1])
        ax.set_ylim(window[2], window[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    stats = (
        f"SJ {metrics.get('scaled_jacobian_min', float('nan')):.3f} | "
        f"ang {metrics.get('min_angle_deg', float('nan')):.1f} | "
        f"AR {metrics.get('aspect_ratio_max', float('nan')):.2f} | "
        f"{metrics.get('n_quads', '?')}q/{metrics.get('n_triangles', '?')}t"
    )
    ax.set_title(f"{title}\n{stats}", fontsize=8)


def draw_failed_panel(ax: plt.Axes, title: str, reason: str) -> None:
    ax.set_facecolor("0.92")
    ax.text(0.5, 0.5, reason, ha="center", va="center", fontsize=9, color="#b33", wrap=True, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8)


def render_board(
    board_id: str,
    panels: list[dict[str, Any]],
    zoom: tuple[float, float, float, float] | None,
    out_name: str,
) -> Path:
    rows = 2 if zoom else 1
    fig, axes = plt.subplots(rows, len(panels), figsize=(3.1 * len(panels), 3.4 * rows), squeeze=False)
    for col, panel in enumerate(panels):
        ax = axes[0][col]
        if panel.get("failed"):
            draw_failed_panel(ax, panel["title"], panel["failed"])
        else:
            draw_panel(ax, panel["mesh"], panel["title"], panel["metrics"], None)
        if rows == 2:
            az = axes[1][col]
            if panel.get("failed"):
                draw_failed_panel(az, f"{panel['title']} (zoom)", panel["failed"])
            else:
                draw_panel(az, panel["mesh"], f"{panel['title']} (zoom)", panel["metrics"], zoom)
    fig.suptitle(f"{board_id}: certified recipes (t=1.0) vs live glm-5.3 agent (t=2.0)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95 if rows == 2 else 0.92))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / out_name
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def certified(board: str, recipe: str) -> Path:
    return ROOT / "workspace" / "block_sweep" / board / recipe


def agent_session(pattern: str) -> Path:
    return sorted((ROOT / "workspace").glob(pattern))[-1]


def main() -> int:
    panels_h00 = []
    for recipe, label in (("baseline", "L0 baseline"), ("ogrid", "L2 ogrid"), ("block", "L3 block")):
        d = certified("train-holes-00", recipe)
        panels_h00.append(
            {"title": f"{label} (t=1.0)", "mesh": d / "mesh.msh", "metrics": metrics_for(d, d / "mesh.msh")}
            if (d / "mesh.msh").is_file()
            else {"title": f"{label} (t=1.0)", "failed": "no mesh in sweep"}
        )
    for label, pattern in (
        ("AGENT own blend v2 (t=2.0, clean)", "train-holes-00-20260916-185130"),
        ("AGENT own blend v1 (t=2.0, clean, cut)", "train-holes-00-20260916-162312"),
        ("AGENT recipe-reuse (t=2.0, leak)", "train-holes-00-20260916-141349"),
    ):
        d = agent_session(pattern)
        mesh = d / "mesh.msh"
        if mesh.is_file():
            panels_h00.append({"title": label, "mesh": mesh, "metrics": metrics_for(d, mesh)})
        else:
            panels_h00.append({"title": label, "failed": "no mesh produced"})
    p1 = render_board(
        "train-holes-00",
        panels_h00,
        zoom=(0.0, 9.0, 0.0, 9.0),
        out_name="train-holes-00_levels_vs_agent.png",
    )
    print("wrote", p1)

    panels_s01 = []
    for recipe, label in (("baseline", "L0 baseline"), ("ogrid", "L2 ogrid")):
        d = certified("train-slots-01", recipe)
        panels_s01.append({"title": f"{label} (t=1.0)", "mesh": d / "mesh.msh", "metrics": metrics_for(d, d / "mesh.msh")})
    d3 = certified("train-slots-01", "block")
    panels_s01.append(
        {"title": "L3 block (t=1.0)", "failed": "fails closed:\nchamfer lattice\nnon-conformant"}
        if not (d3 / "mesh.msh").is_file()
        else {"title": "L3 block (t=1.0)", "mesh": d3 / "mesh.msh", "metrics": metrics_for(d3, d3 / "mesh.msh")}
    )
    for label, pattern in (
        ("AGENT pocket blocks (t=2.0, clean)", "train-slots-01-20260916-193552"),
        ("AGENT tuned L2 (t=2.0)", "train-slots-01-20260916-145059"),
    ):
        da = agent_session(pattern)
        mesh = da / "mesh.msh"
        panels_s01.append(
            {"title": label, "mesh": mesh, "metrics": metrics_for(da, mesh)}
            if mesh.is_file()
            else {"title": label, "failed": "no mesh produced"}
        )
    p2 = render_board(
        "train-slots-01",
        panels_s01,
        zoom=(0.0, 9.0, 0.0, 9.0),
        out_name="train-slots-01_levels_vs_agent.png",
    )
    print("wrote", p2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
