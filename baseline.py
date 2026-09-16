"""Part 1: a minimal Deep Agent that produces one Level-0 Gmsh mesh."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from deepagents import (
    FilesystemMiddleware,
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.backends import LocalShellBackend
from dotenv import load_dotenv
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    TodoListMiddleware,
    ToolCallLimitMiddleware,
    wrap_tool_call,
)
from langchain_core.messages import AIMessage, ToolMessage, messages_to_dict
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.types import Command

from quadagent import quality

REPO_ROOT = Path(__file__).resolve().parent
BOARDS_DIR = REPO_ROOT / "boards"
WORKSPACE_ROOT = REPO_ROOT / "workspace"
TARGET_EDGE_MM = 2.0
MAX_MODEL_CALLS = 12
MAX_EXECUTE_CALLS = 4
DEFAULT_MODEL = "glm-5.3"
DEFAULT_BASE_URL = "https://api.z.ai/api/coding/paas/v4"

BOARDS = {
    "holes": BOARDS_DIR / "train-holes-00.json",
    "rects": BOARDS_DIR / "train-rects-00.json",
    "slots": BOARDS_DIR / "train-slots-00.json",
    "dense": BOARDS_DIR / "train-dense-00.json",
}

REQUIRED_ARTIFACTS = (
    "mesh.py",
    "mesh.msh",
    "mesh.vtk",
    "mesh.png",
    "quality.json",
    "todos.json",
    "transcript.json",
    "run.json",
)

SYSTEM_PROMPT = f"""\
You are QuadAgent Part 1, a deliberately simple meshing agent for a lecture.
Create one valid, first-order, quad-dominant 2D Gmsh mesh from /board.json.

Workflow:
1. Call write_todos once with a short plan.
2. Read /board.json and write the complete program to /mesh.py.
3. The execute tool already starts in the workspace. Run exactly
   `python mesh.py` with a timeout no greater than 120 seconds; never `cd /`
   and do not spend execute calls probing the environment. If the program
   fails, inspect that error, overwrite /mesh.py, and retry. Retries are only
   for execution or missing artifact failures.
4. After a successful execution, call mesh_quality exactly once on /mesh.msh.
   It must be the only tool call in that model response. Never call it after a
   failed execution. mesh_quality is your final action. Do not modify or
   execute anything after it.

The mesh program must:
- Resolve board.json, mesh.msh, mesh.vtk, and mesh.png relative to the directory
  containing mesh.py. A leading slash is virtual only for file tools and must
  not appear in Python paths or shell commands.
- Construct the rectangular outline from its lower-left corner (0, 0) to
  (width, height) in the z=0 plane. Hole coordinates are circle centers.
  Cutout, slot, and header coordinates are rectangle centers, so their lower
  left is (cx-w/2, cy-h/2). Subtract every listed void using one OCC Boolean
  cut, then synchronize. The four active boards have no chamfers.
- Use uniform target size {TARGET_EDGE_MM:.1f} mm everywhere.
- Set Mesh.Algorithm=6, Mesh.RecombineAll=1,
  Mesh.RecombinationAlgorithm=1, and Mesh.Smoothing=10.
- Generate a 2D first-order mesh and write mesh.msh and mesh.vtk.
- Use matplotlib with the Agg backend to write mesh.png. Draw quad edges in
  light gray and fill any remaining triangles red. Use equal axis scaling.
  Gmsh returns flattened coordinates: unpack exactly as
  `node_tags, coords, _ = gmsh.model.mesh.getNodes()` and map each node tag to
  `(coords[3*i], coords[3*i+1])` before finalizing Gmsh.
- Print a short success line with element counts.

This is the baseline, not the improvement stage. Do not use size fields,
local refinement, transfinite curves or surfaces, O-grid construction, block
decomposition, or any quality-driven mesh revision. Never open a GUI.
"""


def resolve_board(alias: str) -> Path:
    """Resolve one of the four lecture board aliases."""
    try:
        return BOARDS[alias]
    except KeyError as exc:
        choices = ", ".join(sorted(BOARDS))
        raise ValueError(f"unknown board {alias!r}; choose one of: {choices}") from exc


def _workspace_path(workspace: Path, virtual_path: str) -> Path:
    candidate = (workspace / virtual_path.lstrip("/")).resolve()
    try:
        candidate.relative_to(workspace.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes workspace: {virtual_path!r}") from exc
    return candidate


def _quality_tool(workspace: Path) -> BaseTool:
    @tool("mesh_quality", return_direct=True)
    def mesh_quality(mesh_file: str = "/mesh.msh") -> str:
        """Score the finished mesh once, save quality.json, and end the run.

        Args:
            mesh_file: Virtual workspace path of the final Gmsh mesh.
        """
        mesh_path = _workspace_path(workspace, mesh_file)
        board = json.loads((workspace / "board.json").read_text())
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "quadagent.quality",
                str(mesh_path),
                "--expected-area",
                str(quality.analytic_area(board)),
            ],
            cwd=workspace,
            env=_shell_environment(),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"mesh scorer failed: {detail}")
        report = json.loads(completed.stdout)
        (workspace / "quality.json").write_text(
            quality.summarize_json(report) + "\n"
        )
        return quality.summarize(report)

    return mesh_quality


@wrap_tool_call
def _terminal_mesh_quality(request: Any, handler: Any) -> Any:
    """Make a successful score jump directly from the tool node to graph end."""
    response = handler(request)
    if request.tool_call["name"] != "mesh_quality":
        return response
    return Command(update={"messages": [response]}, goto=END)


def _shell_environment() -> dict[str, str]:
    """Return the minimal environment exposed to trusted workspace commands."""
    active_python_dir = Path(sys.executable).parent
    return {
        "PATH": os.pathsep.join(
            (str(active_python_dir), "/usr/local/bin", "/usr/bin", "/bin")
        ),
        "MPLBACKEND": "Agg",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def build_agent(
    workspace: Path,
    *,
    api_key: str,
    model_name: str,
    base_url: str,
) -> Any:
    """Build the single-agent Part 1 graph bound to one workspace."""
    workspace = workspace.resolve()
    backend = LocalShellBackend(
        root_dir=workspace,
        virtual_mode=True,
        timeout=120,
        env=_shell_environment(),
        inherit_env=False,
    )
    model = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.2,
        max_retries=2,
        use_responses_api=False,
    )
    register_harness_profile(
        f"openai:{model_name}",
        HarnessProfile(
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)
        ),
    )
    return create_deep_agent(
        model=model,
        tools=[_quality_tool(workspace)],
        system_prompt=SYSTEM_PROMPT,
        backend=backend,
        subagents=[],
        middleware=[
            FilesystemMiddleware(
                backend=backend,
                tools=["read_file", "write_file", "execute"],
                max_execute_timeout=120,
            ),
            TodoListMiddleware(),
            _terminal_mesh_quality,
            ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end"),
            ToolCallLimitMiddleware(
                tool_name="execute",
                run_limit=MAX_EXECUTE_CALLS,
                exit_behavior="continue",
            ),
        ],
        name="quadagent-part-1",
    )


def _new_workspace(board_path: Path, root: Path) -> Path:
    payload = json.loads(board_path.read_text())
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    workspace = root.resolve() / f"{payload['board_id']}-{stamp}"
    workspace.mkdir(parents=True)
    shutil.copyfile(board_path, workspace / "board.json")
    return workspace


def _tool_calls(messages: list[Any]) -> list[str]:
    return [
        call["name"]
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]


def _usage(messages: list[Any]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for message in messages:
        if not isinstance(message, AIMessage) or not message.usage_metadata:
            continue
        for key, value in message.usage_metadata.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals


def _print_message(message: Any) -> None:
    if isinstance(message, AIMessage):
        for call in message.tool_calls:
            print(f"tool: {call['name']}")
    elif isinstance(message, ToolMessage):
        content = str(message.content)
        print(content if message.name == "mesh_quality" else content[:1200])


def _save_state(
    workspace: Path,
    state: dict[str, Any],
    *,
    alias: str,
    model_name: str,
    base_url: str,
    elapsed_s: float,
    error: str | None,
) -> None:
    messages = list(state.get("messages", []))
    calls = _tool_calls(messages)
    usage = _usage(messages)
    (workspace / "transcript.json").write_text(
        json.dumps(messages_to_dict(messages), indent=2, default=str) + "\n"
    )
    (workspace / "todos.json").write_text(
        json.dumps(state.get("todos", []), indent=2, default=str) + "\n"
    )
    payload = {
        "board": alias,
        "board_id": json.loads((workspace / "board.json").read_text())["board_id"],
        "model": model_name,
        "base_url": base_url,
        "elapsed_s": round(elapsed_s, 3),
        "model_calls": sum(isinstance(message, AIMessage) for message in messages),
        "tool_calls": calls,
        "usage": usage,
        "error": error,
    }
    (workspace / "run.json").write_text(json.dumps(payload, indent=2) + "\n")


def run_board(
    alias: str,
    *,
    workspace_root: Path = WORKSPACE_ROOT,
    api_key: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
    emit: bool = True,
) -> Path:
    """Run the real GLM-backed baseline and return its preserved workspace."""
    load_dotenv(REPO_ROOT / ".env")
    key = api_key or os.environ.get("ZAI_API_KEY", "")
    if not key:
        raise RuntimeError("ZAI_API_KEY is not configured")
    selected_model = model_name or os.environ.get("QUADAGENT_MODEL", DEFAULT_MODEL)
    selected_url = base_url or os.environ.get("ZAI_BASE_URL", DEFAULT_BASE_URL)
    board_path = resolve_board(alias)
    workspace = _new_workspace(board_path, workspace_root)
    agent = build_agent(
        workspace,
        api_key=key,
        model_name=selected_model,
        base_url=selected_url,
    )
    task = (
        f"Mesh /board.json ({json.loads(board_path.read_text())['board_id']}) "
        f"at the fixed {TARGET_EDGE_MM:.1f} mm baseline size."
    )
    state: dict[str, Any] = {"messages": []}
    seen = 0
    error: str | None = None
    started = time.monotonic()
    if emit:
        print(f"board: {board_path.name}\nworkspace: {workspace}\nmodel: {selected_model}")
    try:
        for update in agent.stream(
            {"messages": task},
            stream_mode="values",
        ):
            state = update
            messages = list(state.get("messages", []))
            if emit:
                for message in messages[seen:]:
                    _print_message(message)
            seen = len(messages)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _save_state(
            workspace,
            state,
            alias=alias,
            model_name=selected_model,
            base_url=selected_url,
            elapsed_s=time.monotonic() - started,
            error=error,
        )
    return workspace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("board", choices=sorted(BOARDS))
    args = parser.parse_args(argv)
    try:
        workspace = run_board(args.board)
    except Exception as exc:  # noqa: BLE001 - CLI must preserve and report failed runs
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"artifacts: {workspace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
