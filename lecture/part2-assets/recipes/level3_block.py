"""Level-3 meshing script: 100% pure block-structured web-grid (Cubit-style).

Reads board.json from the current working directory and meshes the ENTIRE
domain with transfinite-recombined quadrilateral blocks: zero
unstructured surfaces, no multi-loop plane surfaces, no boolean ops, no
valence-3/5 irregular nodes. Grid lines flow continuously across blocks.

Algorithm (web-grid multi-block):

1. Hole boxes: per circular hole a square box [cx+-b, cy+-b] with
   b = r + ring, ring = clamp(0.8*clearance, 0.5t, 1.2t) additionally
   capped so the box stays inside the chamfered outline. Box edges within
   0.75t of a wall, a rect-void edge or another box edge snap exactly
   onto it. Post-snap validation: circle margin >= 0.35t, no 2D overlap
   with any void or other box.
2. Global cuts: xs/ys = walls + void edges + box edges + chamfer tangent
   coords, spanning the full domain, deduped and sorted. Hole boxes are
   OPAQUE lattice regions: a foreign cut terminating on a box edge
   subdivides that edge into a collinear chain of lattice segments which
   the quadrant patches absorb as multi-curve transfinite sides. A cut
   falling strictly inside a chamfer corner square breaks the lattice and
   fails the board (falls back to Level 2).
3. Cell classification: VOID (inside a rect void / header), BOX (inside
   a hole box) or SOLID; every SOLID cell becomes one 4-sided transfinite
   rectangle patch.
4. Chamfer absorption: each corner triangle merges with its poleward
   neighbor cell (north/south first, then east/west) into one 4-sided
   patch whose diagonal side is the chamfer edge.
5. Interval law: horizontal segments in column band i share H[i],
   vertical segments in row band j share V[j]; hole arcs take the box
   edge chain totals (A_ns / A_ew); connectors take
   R = max(2, round(ring/t)+1). Opposite-side node totals of every
   patch are asserted equal before meshing. Counts include endpoints.

Outputs (cwd): mesh.msh, mesh.vtk, mesh.png. Validation failures raise
SystemExit with a clear message so sweeps can record them per board.
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
RING_FLOOR = 0.5
RING_CAP = 1.2
RING_CLEARANCE_SAFETY = 0.8
SNAP_REACH = 0.75
MARGIN_MIN = 0.35
DEDUPE_TOL = 1e-6
THIN_BAND = 0.5
GEOM_EPS = 1e-6
SOLID = "solid"
VOID = "void"
BOX = "box"
CORNER = "corner"
CONSUMED = "consumed"


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
    s = max(0.0, min(1.0, ((px - a[0]) * dx + (py - a[1]) * dy) / length_sq))
    return math.hypot(px - (a[0] + s * dx), py - (a[1] + s * dy))


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


def dedupe_sorted(values: list[float], tol: float) -> list[float]:
    out: list[float] = []
    for v in sorted(values):
        if not out or v - out[-1] > tol:
            out.append(v)
    return out


def boxes_overlap(a: dict, b: dict, eps: float = GEOM_EPS) -> bool:
    return (
        min(a["x1"], b["x1"]) - max(a["x0"], b["x0"]) > eps
        and min(a["y1"], b["y1"]) - max(a["y0"], b["y0"]) > eps
    )


def rect_contains(outer: dict, x0: float, x1: float, y0: float, y1: float, eps: float) -> bool:
    return (
        outer["x0"] - eps <= x0 and x1 <= outer["x1"] + eps
        and outer["y0"] - eps <= y0 and y1 <= outer["y1"] + eps
    )


def main() -> None:
    board = json.loads(Path("board.json").read_text())
    board_id = board["board_id"]
    outline = board["outline"]
    w_board, h_board = float(outline["width"]), float(outline["height"])
    t = TARGET
    vertices = outline_vertices(outline)
    chamfers = {c["corner"]: float(c["size"]) for c in outline.get("chamfers", []) if float(c["size"]) > 0.0}
    voids_raw = rectangular_voids(board)
    voids = [
        {
            "x0": float(v["cx"]) - float(v["w"]) / 2.0,
            "x1": float(v["cx"]) + float(v["w"]) / 2.0,
            "y0": float(v["cy"]) - float(v["h"]) / 2.0,
            "y1": float(v["cy"]) + float(v["h"]) / 2.0,
        }
        for v in voids_raw
    ]
    holes = board.get("holes", [])
    warns: list[str] = []

    def fail(msg: str) -> None:
        raise SystemExit(f"blockstructured: board {board_id}: {msg}")

    boxes: list[dict] = []
    for idx, hole in enumerate(holes):
        cx, cy, r = float(hole["cx"]), float(hole["cy"]), float(hole["r"])
        clearance = hole_clearance(board, vertices, hole)
        ring = min(RING_CAP * t, max(RING_FLOOR * t, RING_CLEARANCE_SAFETY * clearance))
        b_max = min(cx, w_board - cx, cy, h_board - cy)
        if "sw" in chamfers:
            b_max = min(b_max, (cx + cy - chamfers["sw"]) / 2.0)
        if "se" in chamfers:
            b_max = min(b_max, (w_board - cx + cy - chamfers["se"]) / 2.0)
        if "ne" in chamfers:
            b_max = min(b_max, (w_board - cx + h_board - cy - chamfers["ne"]) / 2.0)
        if "nw" in chamfers:
            b_max = min(b_max, (cx + h_board - cy - chamfers["nw"]) / 2.0)
        ring = min(ring, b_max - r)
        if ring < MARGIN_MIN * t:
            fail(
                f"hole {idx} at ({cx:.3f}, {cy:.3f}) r={r}: ring {ring:.3f} mm < "
                f"{MARGIN_MIN}t after clearance cap (clearance {clearance:.3f} mm, "
                f"chamfer cap b_max {b_max:.3f} mm); no admissible hole box"
            )
        b = r + ring
        boxes.append(
            {
                "hole": idx,
                "cx": cx,
                "cy": cy,
                "r": r,
                "ring": ring,
                "x0": cx - b,
                "x1": cx + b,
                "y0": cy - b,
                "y1": cy + b,
            }
        )

    void_targets_x = [0.0, w_board] + [c for v in voids for c in (v["x0"], v["x1"])]
    void_targets_y = [0.0, h_board] + [c for v in voids for c in (v["y0"], v["y1"])]

    def snap(value: float, targets: list[float]) -> float:
        if not targets:
            return value
        best = min(targets, key=lambda c: abs(c - value))
        return best if abs(best - value) <= SNAP_REACH * t else value

    for box in boxes:
        box["x0"] = snap(box["x0"], void_targets_x)
        box["x1"] = snap(box["x1"], void_targets_x)
        box["y0"] = snap(box["y0"], void_targets_y)
        box["y1"] = snap(box["y1"], void_targets_y)
    for _ in range(2):
        for box in boxes:
            others_x = [c for o in boxes if o is not box for c in (o["x0"], o["x1"])]
            others_y = [c for o in boxes if o is not box for c in (o["y0"], o["y1"])]
            box["x0"] = snap(box["x0"], others_x)
            box["x1"] = snap(box["x1"], others_x)
            box["y0"] = snap(box["y0"], others_y)
            box["y1"] = snap(box["y1"], others_y)

    def box_is_valid(box: dict) -> bool:
        if box["x0"] < -GEOM_EPS or box["x1"] > w_board + GEOM_EPS:
            return False
        if box["y0"] < -GEOM_EPS or box["y1"] > h_board + GEOM_EPS:
            return False
        for corner, csize in chamfers.items():
            sx0, sx1 = (w_board - csize, w_board) if corner in ("se", "ne") else (0.0, csize)
            sy0, sy1 = (h_board - csize, h_board) if corner in ("ne", "nw") else (0.0, csize)
            if min(box["x1"], sx1) - max(box["x0"], sx0) > GEOM_EPS and min(box["y1"], sy1) - max(box["y0"], sy0) > GEOM_EPS:
                if corner == "sw" and box["x0"] + box["y0"] < csize - GEOM_EPS:
                    return False
                if corner == "se" and (w_board - box["x1"]) + box["y0"] < csize - GEOM_EPS:
                    return False
                if corner == "ne" and (w_board - box["x1"]) + (h_board - box["y1"]) < csize - GEOM_EPS:
                    return False
                if corner == "nw" and box["x0"] + (h_board - box["y1"]) < csize - GEOM_EPS:
                    return False
        for v in voids:
            if boxes_overlap(box, v):
                return False
        return True

    for box in boxes:
        for _ in range(2):
            cx, cy = box["cx"], box["cy"]
            m_w, m_e = cx - box["x0"], box["x1"] - cx
            m_s, m_n = cy - box["y0"], box["y1"] - cy
            target = max(m_w, m_e, m_s, m_n)
            grew = False
            for edge, desired in (
                ("x0", cx - target),
                ("x1", cx + target),
                ("y0", cy - target),
                ("y1", cy + target),
            ):
                candidate = dict(box)
                candidate[edge] = desired
                if abs(desired - box[edge]) < GEOM_EPS or not box_is_valid(candidate):
                    continue
                overlaps = any(
                    boxes_overlap(candidate, other)
                    for other in boxes
                    if other is not box
                )
                if overlaps:
                    continue
                if edge in ("x0", "x1"):
                    candidate[edge] = snap(candidate[edge], void_targets_x)
                else:
                    candidate[edge] = snap(candidate[edge], void_targets_y)
                if not box_is_valid(candidate):
                    continue
                box.update(candidate)
                grew = True
            if not grew:
                break

    for _ in range(2):
        for box in boxes:
            others_x = [c for o in boxes if o is not box for c in (o["x0"], o["x1"])]
            others_y = [c for o in boxes if o is not box for c in (o["y0"], o["y1"])]
            box["x0"] = snap(box["x0"], others_x)
            box["x1"] = snap(box["x1"], others_x)
            box["y0"] = snap(box["y0"], others_y)
            box["y1"] = snap(box["y1"], others_y)

    for box in boxes:
        margin = min(box["x1"] - box["cx"], box["cx"] - box["x0"], box["y1"] - box["cy"], box["cy"] - box["y0"]) - box["r"]
        if margin < MARGIN_MIN * t - GEOM_EPS:
            fail(
                f"hole {box['hole']} box [{box['x0']:.3f},{box['x1']:.3f}]x"
                f"[{box['y0']:.3f},{box['y1']:.3f}] margin {margin:.3f} mm < {MARGIN_MIN}t after snapping"
            )
        if box["x0"] < -GEOM_EPS or box["x1"] > w_board + GEOM_EPS or box["y0"] < -GEOM_EPS or box["y1"] > h_board + GEOM_EPS:
            fail(f"hole {box['hole']} box escapes the outline after snapping")
    for box in boxes:
        for v in voids:
            if boxes_overlap(box, v):
                fail(
                    f"hole {box['hole']} box overlaps a rect void "
                    f"[{v['x0']:.3f},{v['x1']:.3f}]x[{v['y0']:.3f},{v['y1']:.3f}]"
                )
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes_overlap(boxes[i], boxes[j]):
                fail(f"hole box {i} overlaps hole box {j}")

    xs_raw = {0.0, w_board}
    ys_raw = {0.0, h_board}
    for v in voids:
        xs_raw.update((v["x0"], v["x1"]))
        ys_raw.update((v["y0"], v["y1"]))
    for box in boxes:
        xs_raw.update((box["x0"], box["x1"]))
        ys_raw.update((box["y0"], box["y1"]))
    for corner, c in chamfers.items():
        if corner in ("sw", "nw"):
            xs_raw.add(c)
        else:
            xs_raw.add(w_board - c)
        if corner in ("sw", "se"):
            ys_raw.add(c)
        else:
            ys_raw.add(h_board - c)
    xs = dedupe_sorted(list(xs_raw), DEDUPE_TOL)
    ys = dedupe_sorted(list(ys_raw), DEDUPE_TOL)

    for values, axis in ((xs, "x"), (ys, "y")):
        for lo, hi in zip(values, values[1:], strict=False):
            if hi - lo < THIN_BAND * t - GEOM_EPS:
                warns.append(f"thin {axis} band [{lo:.3f},{hi:.3f}] = {hi - lo:.3f} mm < 0.5t (feature edges; accepted)")

    corner_squares: dict[str, tuple[float, float, float, float]] = {}
    for corner, c in chamfers.items():
        sx0, sx1 = (w_board - c, w_board) if corner in ("se", "ne") else (0.0, c)
        sy0, sy1 = (h_board - c, h_board) if corner in ("ne", "nw") else (0.0, c)
        corner_squares[corner] = (sx0, sx1, sy0, sy1)
        for x in xs:
            if sx0 + GEOM_EPS < x < sx1 - GEOM_EPS:
                fail(
                    f"cut x={x:.3f} falls inside the {corner} chamfer corner square "
                    f"[{sx0:.3f},{sx1:.3f}]x[{sy0:.3f},{sy1:.3f}]; the corner triangle "
                    f"cannot be absorbed and the lattice cannot stay conformant "
                    f"(typical cause: hole box edge near the chamfered corner)"
                )
        for y in ys:
            if sy0 + GEOM_EPS < y < sy1 - GEOM_EPS:
                fail(
                    f"cut y={y:.3f} falls inside the {corner} chamfer corner square "
                    f"[{sx0:.3f},{sx1:.3f}]x[{sy0:.3f},{sy1:.3f}]; the corner triangle "
                    f"cannot be absorbed and the lattice cannot stay conformant "
                    f"(typical cause: hole box edge near the chamfered corner)"
                )

    def x_index(value: float) -> int:
        for i, x in enumerate(xs):
            if abs(x - value) <= DEDUPE_TOL * 10:
                return i
        raise AssertionError(f"x={value} not a cut")

    def y_index(value: float) -> int:
        for j, y in enumerate(ys):
            if abs(y - value) <= DEDUPE_TOL * 10:
                return j
        raise AssertionError(f"y={value} not a cut")

    nx, ny = len(xs) - 1, len(ys) - 1
    cells = [[SOLID] * ny for _ in range(nx)]
    for i in range(nx):
        for j in range(ny):
            x0, x1, y0, y1 = xs[i], xs[i + 1], ys[j], ys[j + 1]
            for v in voids:
                if rect_contains(v, x0, x1, y0, y1, GEOM_EPS):
                    cells[i][j] = VOID
                    break
            if cells[i][j] == SOLID:
                for box in boxes:
                    if rect_contains(box, x0, x1, y0, y1, GEOM_EPS):
                        cells[i][j] = BOX
                        break

    absorption: list[dict] = []
    for corner, (sx0, sx1, sy0, sy1) in corner_squares.items():
        i0, j0 = x_index(sx0), y_index(sy0)
        if x_index(sx1) != i0 + 1 or y_index(sy1) != j0 + 1:
            fail(f"{corner} chamfer corner square is not a single lattice cell")
        cells[i0][j0] = CORNER
        mirror_x = corner in ("se", "ne")
        mirror_y = corner in ("ne", "nw")
        primary = (i0, j0 - 1) if mirror_y else (i0, j0 + 1)
        secondary = (i0 - 1, j0) if mirror_x else (i0 + 1, j0)
        mode = None
        neighbor = None
        for cand in (primary, secondary):
            ci, cj = cand
            if 0 <= ci < nx and 0 <= cj < ny and cells[ci][cj] == SOLID:
                mode = "north" if cand == primary else "east"
                neighbor = cand
                break
        if mode is None:
            fail(f"{corner} chamfer triangle has no SOLID neighbor to absorb into")
        cells[neighbor[0]][neighbor[1]] = CONSUMED
        absorption.append(
            {
                "corner": corner,
                "i0": i0,
                "j0": j0,
                "ni": neighbor[0],
                "nj": neighbor[1],
                "mode": mode,
                "mirror_x": mirror_x,
                "mirror_y": mirror_y,
            }
        )

    h_counts = [max(2, int((xs[i + 1] - xs[i]) / t + 0.5) + 1) for i in range(nx)]
    v_counts = [max(2, int((ys[j + 1] - ys[j]) / t + 0.5) + 1) for j in range(ny)]

    for box in boxes:
        ix0, ix1 = x_index(box["x0"]), x_index(box["x1"])
        jy0, jy1 = y_index(box["y0"]), y_index(box["y1"])
        box_w, box_h = box["x1"] - box["x0"], box["y1"] - box["y0"]
        a_ns = max(math.ceil(math.pi * box["r"] / 2.0 / t), math.ceil(box_w / t)) + 1
        a_ew = max(math.ceil(math.pi * box["r"] / 2.0 / t), math.ceil(box_h / t)) + 1
        total = sum(h_counts[i] for i in range(ix0, ix1)) - (ix1 - ix0 - 1)
        if total < a_ns:
            widest = max(range(ix0, ix1), key=lambda i: xs[i + 1] - xs[i])
            h_counts[widest] += a_ns - total
        total = sum(v_counts[j] for j in range(jy0, jy1)) - (jy1 - jy0 - 1)
        if total < a_ew:
            widest = max(range(jy0, jy1), key=lambda j: ys[j + 1] - ys[j])
            v_counts[widest] += a_ew - total

    gmsh.initialize()
    try:
        gmsh.model.add(board_id)
        point_cache: dict[tuple[float, float], int] = {}
        line_cache: dict[tuple[int, int], int] = {}
        set_counts: dict[int, int] = {}
        n_patches = 0
        n_cell_patches = 0
        n_quadrants = 0
        n_merged = 0

        def point(x: float, y: float) -> int:
            key = (round(x, 9), round(y, 9))
            if key not in point_cache:
                point_cache[key] = gmsh.model.geo.addPoint(x, y, 0.0)
            return point_cache[key]

        def seg(a: tuple[float, float], b: tuple[float, float]) -> int:
            pa, pb = point(*a), point(*b)
            key = tuple(sorted((pa, pb)))
            if key not in line_cache:
                line_cache[key] = gmsh.model.geo.addLine(key[0], key[1])
            tag = line_cache[key]
            return tag if key == (pa, pb) else -tag

        def register(signed_tag: int, count: int) -> None:
            tag = abs(signed_tag)
            if tag in set_counts and set_counts[tag] != count:
                raise AssertionError(f"line {tag} needs conflicting transfinite counts {set_counts[tag]} vs {count}")
            set_counts[tag] = count
            gmsh.model.geo.mesh.setTransfiniteCurve(tag, count)

        def make_patch(loop_sides: list[int], corners_xy: list[tuple[float, float]]) -> None:
            loop = gmsh.model.geo.addCurveLoop(loop_sides)
            surface = gmsh.model.geo.addPlaneSurface([loop])
            corner_tags = [point(*c) for c in corners_xy]
            gmsh.model.geo.mesh.setTransfiniteSurface(surface, "Left", corner_tags)
            gmsh.model.geo.mesh.setRecombine(2, surface)

        def band_h(xa: float, xb: float) -> int:
            lo, hi = sorted((xa, xb))
            i = x_index(lo)
            assert abs(xs[i + 1] - hi) <= DEDUPE_TOL * 10, "horizontal segment not one lattice band"
            return h_counts[i]

        def band_v(ya: float, yb: float) -> int:
            lo, hi = sorted((ya, yb))
            j = y_index(lo)
            assert abs(ys[j + 1] - hi) <= DEDUPE_TOL * 10, "vertical segment not one lattice band"
            return v_counts[j]

        for i in range(nx):
            for j in range(ny):
                if cells[i][j] != SOLID:
                    continue
                x0, x1, y0, y1 = xs[i], xs[i + 1], ys[j], ys[j + 1]
                bl, br, tr, tl = (x0, y0), (x1, y0), (x1, y1), (x0, y1)
                sides = [seg(bl, br), seg(br, tr), seg(tr, tl), seg(tl, bl)]
                register(sides[0], h_counts[i])
                register(sides[1], v_counts[j])
                register(sides[2], h_counts[i])
                register(sides[3], v_counts[j])
                make_patch(sides, [bl, br, tr, tl])
                n_patches += 1
                n_cell_patches += 1

        for box in boxes:
            cx, cy, r = box["cx"], box["cy"], box["r"]
            inner_x = [x for x in xs if box["x0"] + DEDUPE_TOL < x < box["x1"] - DEDUPE_TOL]
            inner_y = [y for y in ys if box["y0"] + DEDUPE_TOL < y < box["y1"] - DEDUPE_TOL]
            xchain = [box["x0"]] + inner_x + [box["x1"]]
            ychain = [box["y0"]] + inner_y + [box["y1"]]
            p_corners = [(box["x0"], box["y0"]), (box["x1"], box["y0"]), (box["x1"], box["y1"]), (box["x0"], box["y1"])]
            center = point(cx, cy)
            q_pts = []
            for pc in p_corners:
                theta = math.atan2(pc[1] - cy, pc[0] - cx)
                q_pts.append(gmsh.model.geo.addPoint(cx + r * math.cos(theta), cy + r * math.sin(theta), 0.0))
            p_tags = [point(*pc) for pc in p_corners]
            thetas = [math.atan2(pc[1] - cy, pc[0] - cx) for pc in p_corners]
            for k in range(4):
                sweep = (thetas[(k + 1) % 4] - thetas[k]) % (2.0 * math.pi)
                if sweep >= math.pi - 1e-9:
                    fail(f"hole {box['hole']}: quadrant arc sweep {math.degrees(sweep):.1f} deg >= 180")

            south = []
            south_total = 0
            for xa, xb in zip(xchain, xchain[1:], strict=False):
                s = seg((xa, box["y0"]), (xb, box["y0"]))
                c = band_h(xa, xb)
                register(s, c)
                south.append(s)
                south_total += c - 1
            south_total += 1
            north = []
            north_total = 0
            for xa, xb in zip(xchain, xchain[1:], strict=False):
                s = seg((xb, box["y1"]), (xa, box["y1"]))
                c = band_h(xa, xb)
                register(s, c)
                north.append(s)
                north_total += c - 1
            north_total += 1
            east = []
            east_total = 0
            for ya, yb in zip(ychain, ychain[1:], strict=False):
                s = seg((box["x1"], ya), (box["x1"], yb))
                c = band_v(ya, yb)
                register(s, c)
                east.append(s)
                east_total += c - 1
            east_total += 1
            west = []
            west_total = 0
            for ya, yb in zip(ychain, ychain[1:], strict=False):
                s = seg((box["x0"], yb), (box["x0"], ya))
                c = band_v(ya, yb)
                register(s, c)
                west.append(s)
                west_total += c - 1
            west_total += 1
            assert south_total == north_total and east_total == west_total

            arcs = [
                gmsh.model.geo.addCircleArc(q_pts[k], center, q_pts[(k + 1) % 4])
                for k in range(4)
            ]
            conns = [gmsh.model.geo.addLine(p_tags[k], q_pts[k]) for k in range(4)]
            max_conn = max(math.hypot(pc[0] - cx, pc[1] - cy) - r for pc in p_corners)
            r_count = max(2, int(max_conn / (0.75 * t) + 0.5) + 1)
            register(arcs[0], south_total)
            register(arcs[2], north_total)
            register(arcs[1], east_total)
            register(arcs[3], west_total)
            for conn in conns:
                register(conn, r_count)

            quadrant_loops = [
                (south + [conns[1], -arcs[0], -conns[0]], [p_tags[0], p_tags[1], q_pts[1], q_pts[0]]),
                (east + [conns[2], -arcs[1], -conns[1]], [p_tags[1], p_tags[2], q_pts[2], q_pts[1]]),
                (north + [conns[3], -arcs[2], -conns[2]], [p_tags[2], p_tags[3], q_pts[3], q_pts[2]]),
                (west + [conns[0], -arcs[3], -conns[3]], [p_tags[3], p_tags[0], q_pts[0], q_pts[3]]),
            ]
            for sides, corner_pts in quadrant_loops:
                loop = gmsh.model.geo.addCurveLoop(sides)
                surface = gmsh.model.geo.addPlaneSurface([loop])
                gmsh.model.geo.mesh.setTransfiniteSurface(surface, "Left", corner_pts)
                gmsh.model.geo.mesh.setRecombine(2, surface)
                n_patches += 1
                n_quadrants += 1

        for spec in absorption:
            corner = spec["corner"]
            c = chamfers[corner]
            mirror_x, mirror_y = spec["mirror_x"], spec["mirror_y"]
            sx0, sx1, sy0, sy1 = corner_squares[corner]
            ni, nj = spec["ni"], spec["nj"]

            def to_real(u: float, v: float) -> tuple[float, float]:
                return (w_board - u if mirror_x else u, h_board - v if mirror_y else v)

            if spec["mode"] == "north":
                rt = ys[max(nj, spec["j0"]) + 1] - ys[min(nj, spec["j0"])]
                a_f, b_f, c_f, d_f, e_f = (c, 0.0), (c, c), (c, rt), (0.0, rt), (0.0, c)
                v_bottom = band_v(*sorted((to_real(0.0, 0.0)[1], to_real(0.0, c)[1])))
                v_top = band_v(*sorted((to_real(0.0, c)[1], to_real(0.0, rt)[1])))
                h_top = band_h(*sorted((to_real(0.0, rt)[0], to_real(c, rt)[0])))
                v_total = v_bottom + v_top - 1
                frame_edges = [
                    (a_f, b_f, v_bottom),
                    (b_f, c_f, v_top),
                    (c_f, d_f, h_top),
                    (d_f, e_f, v_total),
                    (e_f, a_f, h_top),
                ]
                frame_corners = [a_f, c_f, d_f, e_f]
            else:
                xr = xs[max(ni, spec["i0"]) + 1] - xs[min(ni, spec["i0"])]
                a_f, f_f, g_f, b_f, e_f = (c, 0.0), (xr, 0.0), (xr, c), (c, c), (0.0, c)
                v_side = band_v(*sorted((to_real(0.0, 0.0)[1], to_real(0.0, c)[1])))
                h_inner = band_h(*sorted((to_real(0.0, c)[0], to_real(c, c)[0])))
                h_outer = band_h(*sorted((to_real(c, c)[0], to_real(xr, c)[0])))
                h_total = h_inner + h_outer - 1
                frame_edges = [
                    (a_f, f_f, h_total),
                    (f_f, g_f, v_side),
                    (g_f, b_f, h_outer),
                    (b_f, e_f, h_inner),
                    (e_f, a_f, v_side),
                ]
                frame_corners = [a_f, f_f, g_f, e_f]

            real_edges = [(to_real(*e[0]), to_real(*e[1]), e[2]) for e in frame_edges]
            real_corners = [to_real(*p) for p in frame_corners]
            if mirror_x != mirror_y:
                real_edges = [(b, a, cnt) for (a, b, cnt) in reversed(real_edges)]
                real_corners = list(reversed(real_corners))
            sides = []
            for pa, pb, cnt in real_edges:
                s = seg(pa, pb)
                register(s, cnt)
                sides.append(s)
            make_patch(sides, real_corners)
            n_patches += 1
            n_merged += 1

        gmsh.model.geo.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeMin", t)
        gmsh.option.setNumber("Mesh.MeshSizeMax", t)
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), t)
        gmsh.option.setNumber("Mesh.RecombineAll", 0)
        gmsh.option.setNumber("Mesh.Smoothing", 0)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(1)
        surfaces = [tag for _, tag in gmsh.model.getEntities(2)]
        assert surfaces, "empty domain"
        gmsh.model.addPhysicalGroup(2, surfaces, name="material")
        gmsh.write("mesh.msh")
        gmsh.write("mesh.vtk")

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        lookup = {int(tag): (coords[3 * i], coords[3 * i + 1]) for i, tag in enumerate(node_tags)}
        etypes, _, enodes = gmsh.model.mesh.getElements(2)
        quads, tris = [], []
        for etype, conn in zip(etypes, enodes, strict=True):
            for row in zip(*[iter(conn)] * (4 if etype == 3 else 3), strict=False):
                poly = [lookup[int(tag)] for tag in row]
                (quads if len(row) == 4 else tris).append(poly)
    finally:
        gmsh.finalize()

    for msg in warns:
        print(f"warning: {msg}")
    print(
        f"blocks: {n_patches} transfinite patches = {n_cell_patches} solid cells + "
        f"{n_quadrants} hole quadrants ({len(boxes)} holes x 4) + {n_merged} chamfer-merged"
    )
    if tris:
        print(f"WARNING: {len(tris)} triangles leaked")
    print(f"mesh written: {len(quads)} quads, {len(tris)} tris (+{n_patches} blocks)")

    fig, ax = plt.subplots(figsize=(6, 6))
    if quads:
        ax.add_collection(PolyCollection(quads, facecolor="none", edgecolor="0.55", linewidth=0.35))
    if tris:
        ax.add_collection(PolyCollection(tris, facecolor="red", edgecolor="red", alpha=0.6, linewidth=0.35))
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title(f"{board_id}: {len(quads)} quads, {len(tris)} tris (+{n_patches} blocks)", fontsize=10)
    fig.savefig("mesh.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
