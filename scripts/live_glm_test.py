"""Run and validate real GLM-backed Part 1 sessions.

Examples:
    uv run python scripts/live_glm_test.py holes
    uv run python scripts/live_glm_test.py holes rects slots dense
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import baseline  # noqa: E402 - support direct `python scripts/live_glm_test.py`


class ValidationError(RuntimeError):
    """A live workspace does not satisfy the Part 1 contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def validate_workspace(workspace: Path) -> dict[str, Any]:
    """Validate artifacts, API evidence, tool order, and mesh integrity."""
    workspace = workspace.resolve()
    for name in baseline.REQUIRED_ARTIFACTS:
        path = workspace / name
        _require(path.is_file() and path.stat().st_size > 0, f"missing or empty artifact: {name}")

    run = json.loads((workspace / "run.json").read_text())
    report = json.loads((workspace / "quality.json").read_text())
    calls = run.get("tool_calls", [])
    usage = run.get("usage", {})

    _require(run.get("error") is None, f"agent error: {run.get('error')}")
    _require(0 < int(run.get("model_calls", 0)) <= baseline.MAX_MODEL_CALLS, "invalid model-call count")
    _require(int(usage.get("total_tokens", 0)) > 0, "no GLM token usage was recorded")
    execute_count = calls.count("execute")
    _require(1 <= execute_count <= baseline.MAX_EXECUTE_CALLS, "invalid execute-call count")
    _require(calls.count("mesh_quality") == 1, "mesh_quality must be called exactly once")
    _require(calls[-1:] == ["mesh_quality"], "mesh_quality must be the terminal tool call")
    _require(int(report.get("n_elements_2d", 0)) > 0, "mesh has no 2D elements")
    _require(int(report.get("n_quads", 0)) > 0, "mesh has no quads")
    _require(float(report.get("scaled_jacobian_min", 0.0)) > 0.0, "mesh has an invalid scaled Jacobian")
    area_ratio = float(report.get("area_ratio", 0.0))
    _require(0.98 <= area_ratio <= 1.02, f"area ratio outside [0.98, 1.02]: {area_ratio}")
    return {"workspace": str(workspace), "run": run, "quality": report}


def _latest_workspace(alias: str) -> Path | None:
    board_id = json.loads(baseline.resolve_board(alias).read_text())["board_id"]
    matches = sorted(
        baseline.WORKSPACE_ROOT.glob(f"{board_id}-*"),
        key=lambda path: path.stat().st_mtime,
    )
    return matches[-1] if matches else None


def run_live_matrix(aliases: list[str]) -> Path:
    """Run the requested boards sequentially and write one validation summary."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_dir = baseline.WORKSPACE_ROOT / "live-tests" / stamp
    summary_dir.mkdir(parents=True, exist_ok=True)
    outcomes: list[dict[str, Any]] = []
    summary_path = summary_dir / "summary.json"

    for alias in aliases:
        print(f"\n=== live GLM: {alias} ===", flush=True)
        try:
            workspace = baseline.run_board(alias)
            outcome = {"board": alias, "status": "passed", **validate_workspace(workspace)}
        except Exception as exc:  # noqa: BLE001 - preserve evidence and fail the matrix cleanly
            workspace = _latest_workspace(alias)
            outcome = {
                "board": alias,
                "status": "failed",
                "workspace": str(workspace) if workspace else None,
                "error": f"{type(exc).__name__}: {exc}",
            }
            outcomes.append(outcome)
            summary_path.write_text(json.dumps({"outcomes": outcomes}, indent=2) + "\n")
            raise ValidationError(f"{alias} failed: {outcome['error']}") from exc
        outcomes.append(outcome)
        summary_path.write_text(json.dumps({"outcomes": outcomes}, indent=2) + "\n")
        print(f"PASS: {alias} -> {workspace}")
    return summary_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("boards", nargs="+", choices=sorted(baseline.BOARDS))
    args = parser.parse_args(argv)
    try:
        summary = run_live_matrix(args.boards)
    except ValidationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"\nall live GLM checks passed: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
