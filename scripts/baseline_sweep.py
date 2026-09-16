"""Baseline sweep: mesh every board with the Level-0 recipe and build a
self-contained visual HTML report (mesh images + quality gates).

Usage:
    uv run python scripts/baseline_sweep.py [--boards boards] [--out workspace/baseline_sweep]

Every board runs through the same sandboxed pipeline the agent drives
(`quadagent.selftest_mesh`); quality metrics come from `quadagent.quality`.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quadagent import quality
from quadagent.recipes import recipe_source
from quadagent.sandbox import run_python

MESH_SCRIPT = recipe_source("level0_baseline")
TARGET_EDGE_MM = 1.0

CARD_TEMPLATE = """<div class="card {klass}">
  <div class="cardhead"><span class="bid">{bid}</span><span class="fam">{family}</span></div>
  <img src="data:image/png;base64,{img}" alt="{bid}"/>
  <table>
    <tr><td>quads / tris</td><td>{n_quads} / {n_tris}</td><td class="{g_quad}">{s_quad}</td></tr>
    <tr><td>quad fraction</td><td>{quad_fraction}</td><td class="{g_quad}">{s_quad}</td></tr>
    <tr><td>scaled Jacobian min</td><td>{sj_min}</td><td class="{g_sj}">{s_sj}</td></tr>
    <tr><td>aspect ratio max</td><td>{ar_max}</td><td class="{g_ar}">{s_ar}</td></tr>
    <tr><td>min angle</td><td>{min_angle}&deg;</td><td class="{g_ang}">{s_ang}</td></tr>
    <tr><td>elements</td><td>{n_elem}</td><td class="{g_cnt}">{s_cnt}</td></tr>
    <tr><td>h_min / dt_crit</td><td>{h_min} mm / {dt} &micro;s</td><td></td></tr>
    <tr><td>area ratio</td><td>{area_ratio}</td><td></td></tr>
  </table>
</div>"""


def mesh_one(board_path: Path, session: Path, timeout_s: int) -> dict[str, Any]:
    """Run the Level-0 recipe on one board inside its session directory."""
    payload = json.loads(board_path.read_text())
    session.mkdir(parents=True, exist_ok=True)
    (session / "board.json").write_text(board_path.read_text())
    started = time.monotonic()
    result, _script = run_python(MESH_SCRIPT, cwd=session, timeout_s=timeout_s, mode="auto")
    duration = time.monotonic() - started
    entry: dict[str, Any] = {
        "board_id": payload["board_id"],
        "family": payload.get("family", ""),
        "duration_s": round(duration, 1),
        "status": "ok" if result.exit_code == 0 else "failed",
    }
    if result.exit_code != 0:
        entry["error"] = (result.stderr or result.stdout or "").strip().splitlines()[-1][:200]
        return entry
    try:
        report = quality.analyze(
            session / "mesh.msh", expected_area_mm2=quality.analytic_area(payload)
        )
    except Exception as exc:  # noqa: BLE001 - record scorer failures per board
        entry["status"] = "scorer_failed"
        entry["error"] = f"{type(exc).__name__}: {exc}"[:200]
        return entry
    entry["report"] = report
    (session / "quality.json").write_text(quality.summarize_json(report) + "\n")
    return entry


def gate_of(report: dict[str, Any]) -> dict[str, str]:
    """Recompute the five gate badges for one report."""
    from quadagent.quality import GATES, gate_status

    values = {
        "quad": report.get("quad_fraction", 0.0),
        "sj": report.get("scaled_jacobian_min", -1.0),
        "ar": report.get("aspect_ratio_max", float("inf")),
        "ang": report.get("min_angle_deg", 0.0),
        "cnt": report.get("n_elements_2d", 0),
    }
    out = {}
    for key, (metric, better) in {
        "quad": ("quad_fraction", "high"),
        "sj": ("scaled_jacobian_min", "high"),
        "ar": ("aspect_ratio_max", "low"),
        "ang": ("min_angle_deg", "high"),
        "cnt": ("element_count", "count"),
    }.items():
        status = gate_status(metric, float(values[key]), better)
        out[key] = status.upper()
    return out


def build_html(entries: list[dict[str, Any]], out_dir: Path, boards_dir: Path) -> str:
    """Assemble the single-file visual report."""
    ok = [e for e in entries if e["status"] == "ok"]
    failed = [e for e in entries if e["status"] != "ok"]
    n = len(ok)
    quad_frac_avg = sum(e["report"]["quad_fraction"] for e in ok) / n if n else 0
    sj_min = min((e["report"]["scaled_jacobian_min"] for e in ok), default=0)
    sj_avg = sum(e["report"]["scaled_jacobian_min"] for e in ok) / n if n else 0
    dt_min = min((e["report"]["dt_critical_us"] for e in ok), default=0)
    all_gates = [gate_of(e["report"]) for e in ok]
    n_all_pass = sum(1 for g in all_gates if set(g.values()) == {"PASS"})

    cards = []
    for entry in sorted(entries, key=lambda e: e["board_id"]):
        if entry["status"] != "ok":
            cards.append(
                f'<div class="card fail"><div class="cardhead"><span class="bid">'
                f'{html.escape(entry["board_id"])}</span></div>'
                f'<p class="err">{html.escape(entry.get("error", entry["status"]))}</p></div>'
            )
            continue
        report = entry["report"]
        session = out_dir / entry["board_id"]
        img = base64.b64encode((session / "mesh.png").read_bytes()).decode()
        g = gate_of(report)
        cards.append(
            CARD_TEMPLATE.format(
                klass="allok" if set(g.values()) == {"PASS"} else "",
                bid=html.escape(entry["board_id"]),
                family=html.escape(entry.get("family", "")),
                img=img,
                n_quads=report["n_quads"],
                n_tris=report["n_triangles"],
                quad_fraction=report["quad_fraction"],
                sj_min=report["scaled_jacobian_min"],
                ar_max=report["aspect_ratio_max"],
                min_angle=report["min_angle_deg"],
                n_elem=report["n_elements_2d"],
                h_min=report.get("edge_len_min_mm", "-"),
                dt=report.get("dt_critical_us", "-"),
                area_ratio=report.get("area_ratio", "-"),
                g_quad=f"g{g['quad']}", s_quad=g["quad"],
                g_sj=f"g{g['sj']}", s_sj=g["sj"],
                g_ar=f"g{g['ar']}", s_ar=g["ar"],
                g_ang=f"g{g['ang']}", s_ang=g["ang"],
                g_cnt=f"g{g['cnt']}", s_cnt=g["cnt"],
            )
        )

    rows = "".join(
        f"<tr><td>{html.escape(e['board_id'])}</td>"
        f"<td>{html.escape(e.get('family',''))}</td>"
        f"<td>{e['duration_s']}s</td>"
        + "".join(f"<td>{e['report'].get(k, '-')}</td>" for k in (
            "quad_fraction", "scaled_jacobian_min", "scaled_jacobian_mean",
            "aspect_ratio_max", "min_angle_deg", "n_elements_2d",
            "edge_len_min_mm", "dt_critical_us", "area_ratio",
        ))
        + "</tr>"
        for e in sorted(ok, key=lambda e: e["board_id"])
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>QuadAgent Level-0 baseline sweep</title>
<style>
 body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 24px; background: #fafafa; color: #222; }}
 h1 {{ font-size: 22px; }} h2 {{ font-size: 16px; margin-top: 32px; }}
 .meta {{ color: #666; font-size: 13px; }}
 .stats {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }}
 .stat {{ background: #fff; border: 1px solid #e2e2e2; border-radius: 8px; padding: 10px 16px; }}
 .stat b {{ display: block; font-size: 20px; }}
 .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 16px; }}
 .card {{ background: #fff; border: 1px solid #e2e2e2; border-radius: 10px; padding: 12px; }}
 .card.allok {{ border-color: #3aa757; }}
 .card.fail {{ border-color: #d33; }}
 .cardhead {{ display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 6px; }}
 .bid {{ font-weight: 700; font-family: monospace; }}
 .fam {{ color: #888; }}
 .card img {{ width: 100%; border-radius: 6px; background: #fff; }}
 table {{ width: 100%; border-collapse: collapse; font-size: 12px; font-family: monospace; margin-top: 8px; }}
 td {{ padding: 2px 4px; border-top: 1px solid #f0f0f0; }}
 td:last-child {{ text-align: right; font-weight: 700; }}
 .gPASS {{ color: #2c8a3e; }} .gWARN {{ color: #b8860b; }} .gFAIL {{ color: #cc3333; }}
 .err {{ color: #cc3333; font-size: 12px; }}
 .full {{ background: #fff; border: 1px solid #e2e2e2; border-radius: 8px; overflow-x: auto; }}
 .full table {{ font-size: 11px; margin: 0; }} .full td, .full th {{ padding: 4px 8px; white-space: nowrap; }}
 .full th {{ background: #f4f4f4; text-align: left; position: sticky; top: 0; }}
</style></head><body>
<h1>QuadAgent &mdash; Level-0 baseline sweep (visual report)</h1>
<p class="meta">Generated {datetime.now():%Y-%m-%d %H:%M} &middot; recipe: Frontal-Delaunay + Blossom
recombination, target edge {TARGET_EDGE_MM} mm &middot; driver: scripted baseline via the agent
sandbox (no live LLM runs yet &mdash; glm-5.3 requires an API key; the agent loop itself was
exercised offline with a scripted model). Boards: {boards_dir}</p>
<div class="stats">
 <div class="stat"><b>{len(entries)}</b>boards meshed</div>
 <div class="stat"><b>{n_all_pass}/{len(ok)}</b>all gates PASS</div>
 <div class="stat"><b>{quad_frac_avg:.3f}</b>avg quad fraction</div>
 <div class="stat"><b>{sj_min:.3f}</b>worst scaled-Jac min</div>
 <div class="stat"><b>{sj_avg:.3f}</b>avg scaled-Jac min</div>
 <div class="stat"><b>{dt_min:.3f}&micro;s</b>min dt_crit (h_min/c)</div>
 <div class="stat"><b>{len(failed)}</b>failures</div>
</div>
<h2>Per-board meshes</h2>
<div class="grid">{"".join(cards)}</div>
<h2>Full metrics table</h2>
<div class="full"><table>
<tr><th>board</th><th>family</th><th>time</th><th>quad_frac</th><th>SJ min</th><th>SJ mean</th>
<th>AR max</th><th>min angle</th><th>elements</th><th>h_min</th><th>dt_crit</th><th>area ratio</th></tr>
{rows}</table></div>
</body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boards", default="boards", help="directory of board JSONs")
    parser.add_argument("--out", default="reports/baseline_sweep", help="output directory")
    parser.add_argument("--timeout", type=int, default=120, help="per-board sandbox timeout (s)")
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

    entries = []
    for board_path in sorted(boards_dir.glob("*.json")):
        payload = json.loads(board_path.read_text())
        if "board_id" not in payload or "outline" not in payload:
            print(f"skipping {board_path.name} (not a board file)")
            continue
        print(f"meshing {board_path.name} ...", flush=True)
        entry = mesh_one(board_path, out_dir / board_path.stem, args.timeout)
        if entry["status"] == "ok":
            r = entry["report"]
            print(
                f"  ok {r['n_quads']} quads / {r['n_triangles']} tris, "
                f"SJmin {r['scaled_jacobian_min']}, ARmax {r['aspect_ratio_max']}"
            )
        else:
            print(f"  {entry['status']}: {entry.get('error', '')}")
        entries.append(entry)

    report_path = out_dir / "report.html"
    report_path.write_text(build_html(entries, out_dir, boards_dir))
    ok = sum(1 for e in entries if e["status"] == "ok")
    print(f"\n{ok}/{len(entries)} boards meshed -> {report_path}")
    return 0 if ok == len(entries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
