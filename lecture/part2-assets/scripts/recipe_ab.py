"""Recipe comparison sweep: Level-0 baseline vs Level-2 O-grid vs Level-3 block.

Meshes every board with all THREE recipes through the agent sandbox (no
live LLM), scores each with quadagent.quality, and builds a
self-contained HTML report at workspace/block_sweep/report.html with
summary stats (avg + worst per metric, 3-way win counts), per-board
triple image cards with delta tables, hole zoom renders for three
representative boards, and red failure cards carrying the exact
validation message for boards a recipe could not mesh.

Usage:
    uv run python scripts/recipe_ab.py [--boards boards] [--out workspace/block_sweep]
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quadagent import quality
from quadagent.sandbox import run_python

from quadagent.recipes import recipe_path

SRC = recipe_path("level0_baseline").parent
RECIPES: list[tuple[str, str, str]] = [
    ("baseline", "level0_baseline.py", "Level-0 baseline"),
    ("ogrid", "level2_ogrid.py", "Level-2 O-grid"),
    ("block", "level3_block.py", "Level-3 block"),
]
TARGET_EDGE_MM = 1.0
ZOOM_BOARDS = ("train-holes-00", "train-dense-00", "holdout-slots-05")
ZOOM_WINDOW_MM = 6.0

METRICS: list[tuple[str, str, str]] = [
    ("scaled_jacobian_min", "high", "SJ min"),
    ("scaled_jacobian_mean", "high", "SJ mean"),
    ("aspect_ratio_max", "low", "AR max"),
    ("min_angle_deg", "high", "min angle"),
    ("quad_fraction", "high", "quad fraction"),
    ("n_elements_2d", "low", "elements"),
    ("dt_critical_us", "high", "dt_crit"),
]


def mesh_one(board_path: Path, session: Path, script: str, timeout_s: int) -> dict[str, Any]:
    """Run one recipe on one board inside its session directory."""
    payload = json.loads(board_path.read_text())
    session.mkdir(parents=True, exist_ok=True)
    (session / "board.json").write_text(board_path.read_text())
    started = time.monotonic()
    result, _script = run_python(script, cwd=session, timeout_s=timeout_s, mode="auto")
    duration = time.monotonic() - started
    entry: dict[str, Any] = {
        "board_id": payload["board_id"],
        "duration_s": round(duration, 1),
        "status": "ok" if result.exit_code == 0 else "failed",
    }
    if result.exit_code != 0:
        entry["error"] = (result.stderr or result.stdout or "").strip().splitlines()[-1][:300]
        return entry
    try:
        report = quality.analyze(
            session / "mesh.msh", expected_area_mm2=quality.analytic_area(payload)
        )
    except Exception as exc:  # noqa: BLE001 - record scorer failures per board
        entry["status"] = "scorer_failed"
        entry["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return entry
    entry["report"] = report
    (session / "quality.json").write_text(quality.summarize_json(report) + "\n")
    return entry


def load_polygons(mesh_path: Path) -> tuple[list[list[tuple[float, float]]], list[list[tuple[float, float]]]]:
    """Read quad/triangle polygons back from a written mesh file."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.open(str(mesh_path))
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        lookup = {int(t): (coords[3 * i], coords[3 * i + 1]) for i, t in enumerate(node_tags)}
        etypes, _, enodes = gmsh.model.mesh.getElements(2)
        quads, tris = [], []
        for etype, conn in zip(etypes, enodes, strict=True):
            for row in zip(*[iter(conn)] * (4 if etype == 3 else 3), strict=False):
                poly = [lookup[int(t)] for t in row]
                (quads if len(row) == 4 else tris).append(poly)
        return quads, tris
    finally:
        gmsh.clear()
        gmsh.finalize()


def render_zoom(board: dict[str, Any], mesh_path: Path, out_png: Path) -> None:
    """Close-up render around the board's first hole, fixed 6x6 mm window."""
    quads, tris = load_polygons(mesh_path)
    hole = board["holes"][0]
    cx, cy = float(hole["cx"]), float(hole["cy"])
    half = ZOOM_WINDOW_MM / 2.0
    fig, ax = plt.subplots(figsize=(4, 4))
    if quads:
        ax.add_collection(PolyCollection(quads, facecolor="none", edgecolor="0.45", linewidth=0.6))
    if tris:
        ax.add_collection(PolyCollection(tris, facecolor="red", edgecolor="red", alpha=0.6, linewidth=0.6))
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_title(f"{board['board_id']} hole @ ({cx:.1f}, {cy:.1f})", fontsize=9)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def better_value(values: dict[str, float], better: str) -> tuple[str | None, float]:
    """Best recipe name (None on tie) and the best value."""
    flip = -1.0 if better == "low" else 1.0
    scored = {name: flip * float(v) for name, v in values.items()}
    best = max(scored.values())
    winners = [name for name, s in scored.items() if abs(s - best) < 1e-12]
    return (winners[0] if len(winners) == 1 else None), best / flip


def outcome(base: float, other: float, better: str) -> str:
    diff = other - base
    if better == "low":
        diff = -diff
    if diff > 1e-9:
        return "win"
    if diff < -1e-9:
        return "loss"
    return "tie"


def build_html(triples: list[dict[str, Any]], out_dir: Path, boards_dir: Path, zooms: dict[str, dict[str, Path]]) -> str:
    """Assemble the single-file 3-way report."""
    ok = [
        t
        for t in triples
        if all(t[name]["status"] == "ok" for name, _, _ in RECIPES)
    ]
    failed = [t for t in triples if t not in ok]
    n = len(ok)

    summary_rows = []
    wins = {m: {name: 0 for name, _, _ in RECIPES} for m, _, _ in METRICS}
    wins_tie = {m: 0 for m, _, _ in METRICS}
    for key, better, label in METRICS:
        cells = {f"{agg}_{name}": [] for agg in ("avg", "worst") for name, _, _ in RECIPES}
        for triple in ok:
            values = {name: float(triple[name]["report"][key]) for name, _, _ in RECIPES}
            winner, best = better_value(values, better)
            if winner is None:
                wins_tie[key] += 1
            else:
                wins[key][winner] += 1
            for name in values:
                cells[f"avg_{name}"].append(values[name])
                cells[f"worst_{name}"].append(values[name])
        row = f"<tr><td>{label}</td>"
        for name, _, _ in RECIPES:
            avg = sum(cells[f"avg_{name}"]) / n if n else 0
            row += f"<td>{avg:.4f}</td>"
        for name, _, _ in RECIPES:
            worst = min(cells[f"worst_{name}"]) if better == "high" and n else (max(cells[f"worst_{name}"]) if n else 0)
            row += f"<td>{worst}</td>"
        win_txt = " / ".join(f"{name} {wins[key][name]}" for name, _, _ in RECIPES)
        row += f"<td>{win_txt} (+{wins_tie[key]} tied)</td></tr>"
        summary_rows.append(row)

    cards = []
    for triple in sorted(triples, key=lambda t: t["board_id"]):
        bid = html.escape(triple["board_id"])
        session = out_dir / triple["board_id"]
        if triple not in ok:
            err_parts = []
            for name, _, label in RECIPES:
                entry = triple[name]
                if entry["status"] != "ok":
                    err_parts.append(f"<b>{label}</b>: {html.escape(str(entry.get('error', entry['status'])))}")
            cards.append(
                f'<div class="card fail"><div class="cardhead"><span class="bid">{bid}</span>'
                f'<span class="fam">{html.escape(triple.get("family", ""))}</span></div>'
                f'<p class="err">{"<br>".join(err_parts)}</p></div>'
            )
            continue
        reports = {name: triple[name]["report"] for name, _, _ in RECIPES}
        delta_rows = []
        for key, better, label in METRICS:
            vb, vo, vk = (float(reports[name][key]) for name, _, _ in RECIPES)
            d_ob, d_ko = vo - vb, vk - vo
            cls_ob = "good" if (d_ob > 0) == (better == "high") and abs(d_ob) > 1e-9 else ("bad" if abs(d_ob) > 1e-9 else "")
            cls_ko = "good" if (d_ko > 0) == (better == "high") and abs(d_ko) > 1e-9 else ("bad" if abs(d_ko) > 1e-9 else "")
            delta_rows.append(
                f"<tr><td>{label}</td><td>{reports['baseline'][key]}</td><td>{reports['ogrid'][key]}</td>"
                f"<td>{reports['block'][key]}</td>"
                f"<td class='{cls_ob}'>{d_ob:+.4f}</td><td class='{cls_ko}'>{d_ko:+.4f}</td></tr>"
            )
        imgs = "".join(
            f'<figure><img src="data:image/png;base64,{b64(session / name / "mesh.png")}"/>'
            f"<figcaption>{label}</figcaption></figure>"
            for name, _, label in RECIPES
        )
        cards.append(f"""<div class="card">
  <div class="cardhead"><span class="bid">{bid}</span><span class="fam">{html.escape(triple.get('family',''))}</span></div>
  <div class="imgs">{imgs}</div>
  <table><tr><th>metric</th><th>baseline</th><th>ogrid</th><th>block</th><th>&Delta; ogrid</th><th>&Delta; block</th></tr>{''.join(delta_rows)}</table>
</div>""")

    zoom_html = []
    for bid in ZOOM_BOARDS:
        if bid not in zooms:
            continue
        board = json.loads((boards_dir / f"{bid}.json").read_text())
        hole = board["holes"][0]
        figs = "".join(
            f'<figure><img src="data:image/png;base64,{b64(zooms[bid][name])}"/><figcaption>{label}</figcaption></figure>'
            for name, _, label in RECIPES
        )
        zoom_html.append(f"""<div class="zoomcard">
  <h3>{bid} &mdash; hole at ({float(hole['cx']):.2f}, {float(hole['cy']):.2f}), r={float(hole['r'])} mm, {ZOOM_WINDOW_MM}x{ZOOM_WINDOW_MM} mm window</h3>
  <div class="imgs">{figs}</div>
</div>""")

    heads = "".join(f"<th>{label}</th>" for _, _, label in RECIPES)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>QuadAgent recipe sweep: baseline vs O-grid vs block</title>
<style>
 body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 24px; background: #fafafa; color: #222; }}
 h1 {{ font-size: 22px; }} h2 {{ font-size: 16px; margin-top: 32px; }} h3 {{ font-size: 13px; }}
 .meta {{ color: #666; font-size: 13px; max-width: 900px; }}
 .grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; max-width: 1100px; }}
 .card, .zoomcard {{ background: #fff; border: 1px solid #e2e2e2; border-radius: 10px; padding: 12px; }}
 .card.fail {{ border-color: #d33; }}
 .cardhead {{ display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 6px; }}
 .bid {{ font-weight: 700; font-family: monospace; }}
 .fam {{ color: #888; }}
 .imgs {{ display: flex; gap: 10px; }}
 .imgs figure {{ flex: 1; margin: 0; }}
 .imgs img {{ width: 100%; border-radius: 6px; background: #fff; border: 1px solid #f0f0f0; }}
 figcaption {{ font-size: 11px; color: #888; text-align: center; margin-top: 2px; }}
 table {{ width: 100%; border-collapse: collapse; font-size: 12px; font-family: monospace; margin-top: 8px; }}
 td, th {{ padding: 3px 6px; border-top: 1px solid #f0f0f0; text-align: right; }}
 th:first-child, td:first-child {{ text-align: left; }}
 td.good {{ color: #2c8a3e; font-weight: 700; }} td.bad {{ color: #cc3333; font-weight: 700; }}
 .err {{ color: #cc3333; font-size: 12px; }}
 .full {{ background: #fff; border: 1px solid #e2e2e2; border-radius: 8px; overflow-x: auto; max-width: 1100px; }}
 .full table {{ font-size: 12px; margin: 0; }}
 .zooms {{ display: grid; grid-template-columns: 1fr; gap: 18px; max-width: 900px; }}
</style></head><body>
<h1>QuadAgent &mdash; recipe sweep: Level-0 baseline vs Level-2 O-grid vs Level-3 block-structured</h1>
<p class="meta">Generated {datetime.now():%Y-%m-%d %H:%M} &middot; target edge {TARGET_EDGE_MM} mm &middot;
driver: scripted recipes (quadagent.selftest_mesh / quadagent.ogrid_mesh / quadagent.blockstructured_mesh)
executed through the agent sandbox &mdash; no live LLM in this sweep. Level 3 fails closed with a recorded
validation message on boards whose lattice cannot stay conformant (chamfered corners with nearby hole boxes);
those boards fall back to Level 2. Boards: {boards_dir}</p>
<h2>Summary ({n} boards meshed by all three; avg / worst per metric; worst = min for higher-better, max for lower-better)</h2>
<div class="full"><table>
<tr><th rowspan="2">metric</th><th colspan="3">avg</th><th colspan="3">worst</th><th rowspan="2">board wins</th></tr>
<tr>{heads}{heads}</tr>
{"".join(summary_rows)}</table></div>
<h2>Hole zooms ({len(zoom_html)} boards)</h2>
<div class="zooms">{''.join(zoom_html)}</div>
<h2>Per-board comparison</h2>
<div class="grid">{''.join(cards)}</div>
<p class="meta">{len(failed)} board(s) not meshed by all three recipes of {len(triples)}.</p>
</body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boards", default="boards", help="directory of board JSONs")
    parser.add_argument("--out", default="reports/block_sweep", help="output directory")
    parser.add_argument("--timeout", type=int, default=180, help="per-run sandbox timeout (s)")
    args = parser.parse_args()

    boards_dir = Path(args.boards)
    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    shared_cache = (out_dir / "_mplcache").resolve()
    shared_cache.mkdir()
    import os

    os.environ["MPLCONFIGDIR"] = str(shared_cache)

    scripts = {name: (SRC / fname).read_text() for name, fname, _ in RECIPES}
    triples: list[dict[str, Any]] = []
    for board_path in sorted(boards_dir.glob("*.json")):
        payload = json.loads(board_path.read_text())
        if "board_id" not in payload or "outline" not in payload:
            print(f"skipping {board_path.name} (not a board file)")
            continue
        bid = payload["board_id"]
        triple: dict[str, Any] = {"board_id": bid, "family": payload.get("family", "")}
        for name, _, label in RECIPES:
            print(f"meshing {bid}: {name} ...", flush=True)
            triple[name] = mesh_one(board_path, out_dir / bid / name, scripts[name], args.timeout)
            entry = triple[name]
            if entry["status"] == "ok":
                r = entry["report"]
                print(
                    f"  {label}: SJmin {r['scaled_jacobian_min']}, ARmax {r['aspect_ratio_max']}, "
                    f"qf {r['quad_fraction']}, nel {r['n_elements_2d']}, dt {r['dt_critical_us']}"
                )
            else:
                print(f"  {label}: {entry['status']}: {entry.get('error', '')}")
        triples.append(triple)

    zooms: dict[str, dict[str, Path]] = {}
    for bid in ZOOM_BOARDS:
        triple = next((t for t in triples if t["board_id"] == bid), None)
        if triple is None or any(triple[name]["status"] != "ok" for name, _, _ in RECIPES):
            continue
        board = json.loads((boards_dir / f"{bid}.json").read_text())
        zooms[bid] = {}
        for name, _, _ in RECIPES:
            png = out_dir / bid / f"zoom_{name}.png"
            render_zoom(board, out_dir / bid / name / "mesh.msh", png)
            zooms[bid][name] = png

    report_path = out_dir / "report.html"
    report_path.write_text(build_html(triples, out_dir, boards_dir, zooms))

    ok = [t for t in triples if all(t[name]["status"] == "ok" for name, _, _ in RECIPES)]
    print(f"\n{len(ok)}/{len(triples)} boards meshed by all three recipes -> {report_path}")
    if ok:
        header = "metric".ljust(20) + "".join(f"{('avg ' + name):>12s}/{('wrst ' + name):>12s}" for name, _, _ in RECIPES) + "  wins (b/o/k)"
        print(header)
        for key, better, label in METRICS:
            cols = []
            values = {name: [float(t[name]["report"][key]) for t in ok] for name, _, _ in RECIPES}
            for name, _, _ in RECIPES:
                avg = sum(values[name]) / len(ok)
                worst = min(values[name]) if better == "high" else max(values[name])
                cols.append(f"{avg:12.4f}/{worst:12.4f}")
            tally = {name: 0 for name, _, _ in RECIPES}
            ties = 0
            for i in range(len(ok)):
                vals = {name: values[name][i] for name, _, _ in RECIPES}
                winner, _best = better_value(vals, better)
                if winner is None:
                    ties += 1
                else:
                    tally[winner] += 1
            print(
                label.ljust(20)
                + "".join(cols)
                + f"  {tally['baseline']}/{tally['ogrid']}/{tally['block']} (+{ties}t)"
            )
    return 0 if len(ok) == len(triples) else 1


if __name__ == "__main__":
    raise SystemExit(main())
