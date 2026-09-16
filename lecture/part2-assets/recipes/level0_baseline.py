"""Known-good Level-0 meshing script used by `quadagent selftest` and tests.

Builds the board domain from board.json with the OCC kernel, applies the
playbook baseline recipe (Frontal-Delaunay + Blossom recombination), and
writes mesh.msh, mesh.vtk and mesh.png into the current directory. The
agent is expected to (re)derive this itself; this file proves the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

import gmsh


def main() -> None:
    board = json.loads(Path("board.json").read_text())
    outline = board["outline"]
    target = 1.0

    gmsh.initialize()
    try:
        gmsh.model.add(board["board_id"])
        w, h = float(outline["width"]), float(outline["height"])
        board_tag = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, w, h)

        tools = []
        for hole in board.get("holes", []):
            tag = gmsh.model.occ.addDisk(
                float(hole["cx"]), float(hole["cy"]), 0.0,
                float(hole["r"]), float(hole["r"]),
            )
            tools.append((2, tag))
        for key in ("cutouts", "slots"):
            for rect in board.get(key, []):
                tag = gmsh.model.occ.addRectangle(
                    float(rect["cx"]) - float(rect["w"]) / 2.0,
                    float(rect["cy"]) - float(rect["h"]) / 2.0,
                    0.0,
                    float(rect["w"]), float(rect["h"]),
                )
                tools.append((2, tag))
        header = board.get("header")
        if header and float(header["w"]) > 0 and float(header["h"]) > 0:
            tag = gmsh.model.occ.addRectangle(
                float(header["cx"]) - float(header["w"]) / 2.0,
                float(header["cy"]) - float(header["h"]) / 2.0,
                0.0,
                float(header["w"]), float(header["h"]),
            )
            tools.append((2, tag))

        if tools:
            gmsh.model.occ.cut([(2, board_tag)], tools, removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeMin", target)
        gmsh.option.setNumber("Mesh.MeshSizeMax", target)
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), target)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
        gmsh.option.setNumber("Mesh.Smoothing", 10)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(1)
        surfaces = [tag for _, tag in gmsh.model.getEntities(2)]
        assert surfaces, "empty domain after boolean cuts"
        gmsh.model.addPhysicalGroup(2, surfaces, name="material")
        gmsh.write("mesh.msh")
        gmsh.write("mesh.vtk")

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        lookup = {int(t): (coords[3 * i], coords[3 * i + 1]) for i, t in enumerate(node_tags)}
        etypes, _, enodes = gmsh.model.mesh.getElements(2)
        quads, tris = [], []
        for etype, conn in zip(etypes, enodes, strict=True):
            for row in zip(*[iter(conn)] * (4 if etype == 3 else 3), strict=False):
                poly = [lookup[int(t)] for t in row]
                (quads if len(row) == 4 else tris).append(poly)
    finally:
        gmsh.finalize()

    fig, ax = plt.subplots(figsize=(6, 6))
    if quads:
        ax.add_collection(PolyCollection(quads, facecolor="none", edgecolor="0.55", linewidth=0.4))
    if tris:
        ax.add_collection(PolyCollection(tris, facecolor="red", edgecolor="red", alpha=0.6, linewidth=0.4))
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title(
        f"{board['board_id']}: {len(quads)} quads, {len(tris)} tris",
        fontsize=10,
    )
    fig.savefig("mesh.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"mesh written: {len(quads)} quads, {len(tris)} tris")


if __name__ == "__main__":
    main()
