"""Command line interface: run the agent on a board, or offline self-tests.

Examples:
    quadagent boards                      # list bundled boards
    quadagent selftest                    # sandbox + quality, no LLM
    quadagent run boards/train-holes-00.json
    quadagent run boards/train-slots-00.json --target-edge 1.5 --max-steps 40
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from quadagent.agent import DeepAgent
from quadagent.config import AgentConfig, load_config
from quadagent.tools import ToolRegistry

BOARDS_DIR = Path(__file__).resolve().parents[2] / "boards"


def _session_dir(root: Path, board_id: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session = root / f"{board_id}-{stamp}"
    session.mkdir(parents=True, exist_ok=True)
    return session


def cmd_boards(args: argparse.Namespace) -> int:
    pattern = args.filter or ""
    boards = sorted(
        p
        for p in BOARDS_DIR.glob("*.json")
        if pattern in p.name and "board_id" in json.loads(p.read_text())
    )
    for path in boards:
        payload = json.loads(path.read_text())
        n_holes = len(payload.get("holes", []))
        n_voids = (
            len(payload.get("cutouts", [])) + len(payload.get("slots", [])) + 1
        )
        print(f"{path.name:28s} {payload.get('family', ''):20s} holes={n_holes} rect_voids={n_voids}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    board_path = Path(args.board)
    if not board_path.is_file():
        print(f"error: board file not found: {board_path}", file=sys.stderr)
        return 2
    cfg = load_config(
        max_steps=args.max_steps,
        workspace_root=Path(args.workspace),
    )
    if not cfg.api_key:
        print(
            "error: ZAI_API_KEY is not set. Copy .env.example to .env and add your "
            "z.ai coding-plan key (https://docs.z.ai/devpack/quick-start).",
            file=sys.stderr,
        )
        return 2

    payload = json.loads(board_path.read_text())
    board_id = payload.get("board_id", board_path.stem)
    session = _session_dir(cfg.workspace_root, board_id)
    (session / "board.json").write_text(board_path.read_text())

    from quadagent.llm import GlmClient
    from quadagent.prompts import build_task

    registry = ToolRegistry(session, cfg)
    agent = DeepAgent(GlmClient(cfg), registry, max_steps=cfg.max_steps, log=print)

    print(f"board:      {board_path}")
    print(f"session:    {session}")
    print(f"model:      {cfg.model} @ {cfg.base_url}")
    print(f"sandbox:    {cfg.sandbox_mode}, timeout {cfg.sandbox_timeout_s}s")
    print("-" * 60)
    task = build_task(board_id, args.target_edge, cfg.max_steps)
    result = agent.run(task)

    print("-" * 60)
    print(f"stop: {result.stop_reason}  model calls: {result.n_model_calls}  "
          f"tool calls: {result.n_tool_calls}")
    if result.usage:
        print(f"tokens: {result.usage}")
    print("\nfinal report:\n")
    print(result.final_text)
    print(f"\nartifacts in: {session}")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """Offline pipeline check: sandboxed gmsh run -> mesh.vtk/png -> quality."""
    cfg = load_config()
    with tempfile.TemporaryDirectory(prefix="quadagent-selftest-") as tmp:
        workspace = Path(tmp)
        (workspace / "board.json").write_text(
            json.dumps(
                {
                    "board_id": "selftest",
                    "outline": {"width": 10.0, "height": 10.0, "chamfers": []},
                    "holes": [{"cx": 5.0, "cy": 5.0, "r": 1.5}],
                    "cutouts": [],
                    "slots": [],
                    "header": {"cx": 5.0, "cy": 9.0, "w": 4.0, "h": 1.0},
                }
            )
        )
        registry = ToolRegistry(workspace, cfg)
        from quadagent.recipes import recipe_source

        code = recipe_source("level0_baseline")
        out = registry.dispatch("execute_python", json.dumps({"code": code}))
        print(out)
        assert "status: OK" in out, "sandboxed meshing script failed"
        quality_out = registry.dispatch("mesh_quality", json.dumps({"mesh_file": "mesh.msh"}))
        print(quality_out)
        assert "quad_fraction" in quality_out
        assert (workspace / "mesh.vtk").is_file(), "mesh.vtk missing"
        assert (workspace / "mesh.png").is_file(), "mesh.png missing"
    print("\nselftest OK: sandbox -> gmsh -> vtk/png -> quality metrics")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quadagent", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    boards = sub.add_parser("boards", help="list the bundled PCB boards")
    boards.add_argument("--filter", default="", help="substring filter")
    boards.set_defaults(func=cmd_boards)

    run = sub.add_parser("run", help="run the agent on one board")
    run.add_argument("board", help="path to a board JSON file")
    run.add_argument("--target-edge", type=float, default=2.0, help="target element size in mm")
    run.add_argument("--max-steps", type=int, default=30)
    run.add_argument("--workspace", default="workspace")
    run.set_defaults(func=cmd_run)

    selftest = sub.add_parser("selftest", help="offline sandbox+gmsh+quality check")
    selftest.set_defaults(func=cmd_selftest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
