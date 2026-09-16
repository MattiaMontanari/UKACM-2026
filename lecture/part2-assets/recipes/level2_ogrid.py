"""Known-good Level-2 meshing script: O-grid rings around every hole.

Reads board.json from the current working directory, builds a structured
annular quad ring (four transfinite-recombined quadrant surfaces) around
every circular hole, meshes the remainder with the Level-0 recipe
(Frontal-Delaunay + Blossom recombination), and writes mesh.msh, mesh.vtk
and mesh.png. Rectangular cutouts/slots/header stay unstructured: the
O-grid scope is holes only.

Ring sizing per hole (t = target edge length, all mm):

- clearance = min distance from the hole boundary to the outline polygon
  edges, every rectangular void, and every other hole;
- ring thickness = clamp(clearance - 1.0 t, 0.5 t, 1.2 t): the ring must
  leave a meshable channel of about one target edge to every neighbouring
  feature, otherwise the unstructured transition squeezes slivers into
  the channel;
- n_arc = max(3, ceil(pi * outer_r / 2 / (0.7 t))) nodes per quadrant arc
  (outer arc step ~ 0.7 t), shared by the inner and outer arc so the
  quadrants stay balanced; the denser-than-t arcs match the refined
  transition zones near walls, which is where the remaining slivers
  lived at full-t arcs;
- n_radial = max(1, round(thickness / mean tangential step)) with the
  radial progression ratio clamped to [1.0, 2.2].

The remainder is meshed with the Level-0 options plus a Distance +
Threshold background field on the FIXED straight boundaries (outline
polygon and rectangular voids; SizeMin 0.5 t within 0.2 t of a curve,
growing back to t at 1.2 t). Two transition pitfalls this guards
against, found empirically: (1) pinch channels between rectangular
voids must be refined or Blossom leaves ~0.2 scaled-Jacobian slivers
there; (2) refining NEAR THE ARCS would mismatch the fixed arc node
spacing against interior sizes and shear the first quad layer - the
arcs are deliberately excluded from the field.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

import gmsh

TARGET = 1.0
RING_THICKNESS_CAP = 1.2
RING_THICKNESS_FLOOR = 0.5
RING_CHANNEL_GAP = 1.0
RADIAL_PROGRESSION_MAX = 2.2
FIELD_SIZE_MIN = 0.5
FIELD_DIST_MIN = 0.2
FIELD_DIST_MAX = 1.2
ARC_STEP_FACTOR = 0.7


def outline_vertices(outline: dict) -> list[tuple[float, float]]:
    """Outline polygon with chamfers resolved, counter-clockwise, closed."""
    w, h = float(outline["width"]), float(outline["height"])
    by_corner = {c["corner"]: float(c["size"]) for c in outline.get("chamfers", [])}
    sw = by_corner.get("sw", 0.0)
    se = by_corner.get("se", 0.0)
    ne = by_corner.get("ne", 0.0)
    nw = by_corner.get("nw", 0.0)
    candidates = [
        (sw, 0.0),
        (w - se, 0.0),
        (w, se),
        (w, h - ne),
        (w - ne, h),
        (nw, h),
        (0.0, h - nw),
        (0.0, sw),
    ]
    kept: list[tuple[float, float]] = []
    for point in candidates:
        if kept and abs(kept[-1][0] - point[0]) < 1e-12 and abs(kept[-1][1] - point[1]) < 1e-12:
            continue
        kept.append(point)
    if len(kept) > 1 and abs(kept[0][0] - kept[-1][0]) < 1e-12 and abs(kept[0][1] - kept[-1][1]) < 1e-12:
        kept.pop()
    return kept


def point_segment_distance(px: float, py: float, a: tuple[float, float], b: tuple[float, float]) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return math.hypot(px - a[0], py - a[1])
    t = max(0.0, min(1.0, ((px - a[0]) * dx + (py - a[1]) * dy) / length_sq))
    return math.hypot(px - (a[0] + t * dx), py - (a[1] + t * dy))


def rectangular_voids(board: dict) -> list[dict]:
    """All axis-aligned rectangular voids (cutouts, slots, header)."""
    voids = list(board.get("cutouts", [])) + list(board.get("slots", []))
    header = board.get("header")
    if header and float(header["w"]) > 0 and float(header["h"]) > 0:
        voids.append(header)
    return voids


def hole_clearance(board: dict, vertices: list[tuple[float, float]], hole: dict) -> float:
    """Minimum distance from one hole boundary to every other feature."""
    cx, cy, r = float(hole["cx"]), float(hole["cy"]), float(hole["r"])
    clearance = math.inf
    for a, b in zip(vertices, vertices[1:] + vertices[:1], strict=False):
        clearance = min(clearance, point_segment_distance(cx, cy, a, b) - r)
    for other in board.get("holes", []):
        if other is hole:
            continue
        clearance = min(
            clearance,
            math.hypot(cx - float(other["cx"]), cy - float(other["cy"])) - r - float(other["r"]),
        )
    for rect in rectangular_voids(board):
        x0, y0 = float(rect["cx"]) - float(rect["w"]) / 2.0, float(rect["cy"]) - float(rect["h"]) / 2.0
        x1, y1 = float(rect["cx"]) + float(rect["w"]) / 2.0, float(rect["cy"]) + float(rect["h"]) / 2.0
        corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        for a, b in zip(corners, corners[1:] + corners[:1], strict=False):
            clearance = min(clearance, point_segment_distance(cx, cy, a, b) - r)
    return clearance


def build_domain(board: dict, target: float) -> None:
    """Geo-kernel domain: transfinite ring per hole + unstructured remainder."""
    cache: dict[tuple[float, float], int] = {}

    def point(x: float, y: float) -> int:
        key = (round(x, 9), round(y, 9))
        if key not in cache:
            cache[key] = gmsh.model.geo.addPoint(x, y, 0.0)
        return cache[key]

    vertices = outline_vertices(board["outline"])
    fixed_curves = [
        gmsh.model.geo.addLine(point(*a), point(*b))
        for a, b in zip(vertices, vertices[1:] + vertices[:1], strict=False)
    ]
    surface_loops = [gmsh.model.geo.addCurveLoop(fixed_curves)]

    for rect in rectangular_voids(board):
        x0, y0 = float(rect["cx"]) - float(rect["w"]) / 2.0, float(rect["cy"]) - float(rect["h"]) / 2.0
        x1, y1 = float(rect["cx"]) + float(rect["w"]) / 2.0, float(rect["cy"]) + float(rect["h"]) / 2.0
        corners = [point(x0, y0), point(x1, y0), point(x1, y1), point(x0, y1)]
        lines = [gmsh.model.geo.addLine(corners[i], corners[(i + 1) % 4]) for i in range(4)]
        fixed_curves.extend(lines)
        surface_loops.append(gmsh.model.geo.addCurveLoop(lines))

    for hole in board.get("holes", []):
        cx, cy, r = float(hole["cx"]), float(hole["cy"]), float(hole["r"])
        clearance = hole_clearance(board, vertices, hole)
        thickness = max(
            RING_THICKNESS_FLOOR * target,
            min(RING_THICKNESS_CAP * target, clearance - RING_CHANNEL_GAP * target),
        )
        outer_r = r + thickness
        n_arc = max(3, math.ceil((math.pi * outer_r / 2.0) / (ARC_STEP_FACTOR * target)))
        inner_step = (math.pi * r / 2.0) / n_arc
        outer_step = (math.pi * outer_r / 2.0) / n_arc
        n_radial = max(1, round(thickness / max(1e-9, (inner_step + outer_step) / 2.0)))
        progression = min(RADIAL_PROGRESSION_MAX, max(1.0, outer_step / max(1e-9, inner_step)))

        center = point(cx, cy)
        inner_pts = [point(cx + r, cy), point(cx, cy + r), point(cx - r, cy), point(cx, cy - r)]
        outer_pts = [point(cx + outer_r, cy), point(cx, cy + outer_r), point(cx - outer_r, cy), point(cx, cy - outer_r)]
        inner_arcs = [gmsh.model.geo.addCircleArc(inner_pts[k], center, inner_pts[(k + 1) % 4]) for k in range(4)]
        outer_arcs = [gmsh.model.geo.addCircleArc(outer_pts[k], center, outer_pts[(k + 1) % 4]) for k in range(4)]
        radials = [gmsh.model.geo.addLine(inner_pts[k], outer_pts[k]) for k in range(4)]
        for k in range(4):
            loop = gmsh.model.geo.addCurveLoop(
                [radials[k], outer_arcs[k], -radials[(k + 1) % 4], -inner_arcs[k]]
            )
            surface = gmsh.model.geo.addPlaneSurface([loop])
            gmsh.model.geo.mesh.setTransfiniteCurve(inner_arcs[k], n_arc + 1)
            gmsh.model.geo.mesh.setTransfiniteCurve(outer_arcs[k], n_arc + 1)
            if n_radial > 1 and progression > 1.02:
                gmsh.model.geo.mesh.setTransfiniteCurve(radials[k], n_radial + 1, "Progression", progression)
            else:
                gmsh.model.geo.mesh.setTransfiniteCurve(radials[k], n_radial + 1)
            gmsh.model.geo.mesh.setTransfiniteSurface(
                surface,
                "Left",
                [inner_pts[k], outer_pts[k], outer_pts[(k + 1) % 4], inner_pts[(k + 1) % 4]],
            )
            gmsh.model.geo.mesh.setRecombine(2, surface)
        surface_loops.append(gmsh.model.geo.addCurveLoop(outer_arcs))

    gmsh.model.geo.addPlaneSurface(surface_loops)
    gmsh.model.geo.synchronize()

    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.3 * target)
    gmsh.option.setNumber("Mesh.MeshSizeMax", target)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), target)

    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "CurvesList", fixed_curves)
    gmsh.model.mesh.field.setNumber(distance, "Sampling", 100)
    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMin", FIELD_SIZE_MIN * target)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMax", target)
    gmsh.model.mesh.field.setNumber(threshold, "DistMin", FIELD_DIST_MIN * target)
    gmsh.model.mesh.field.setNumber(threshold, "DistMax", FIELD_DIST_MAX * target)
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold)


def main() -> None:
    board = json.loads(Path("board.json").read_text())
    target = TARGET

    gmsh.initialize()
    try:
        gmsh.model.add(board["board_id"])
        build_domain(board, target)

        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
        gmsh.option.setNumber("Mesh.Smoothing", 10)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(1)
        surfaces = [tag for _, tag in gmsh.model.getEntities(2)]
        assert surfaces, "empty domain"
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
