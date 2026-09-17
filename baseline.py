"""Run a small Deep Agent that creates a Gmsh mesh from a board description."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
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
from langchain_core.tools import BaseTool, tool
from langchain_core.tracers.langchain import LangChainTracer, wait_for_all_tracers
from langchain_openai import ChatOpenAI

from quadagent import quality

ROOT = Path(__file__).resolve().parent
WORKSPACES = ROOT / "workspace"
TARGET_SIZE_MM = 1.5
MODEL = "glm-5.3"
BASE_URL = "https://api.z.ai/api/coding/paas/v4"
BOARDS = {
    path.stem.removeprefix("train-").removesuffix("-00"): path
    for path in sorted((ROOT / "boards").glob("*.json"))
}

SYSTEM_PROMPT = f"""\
Read /board.json and create a first-order, 2D, quad-dominant Gmsh mesh.
Write the complete generator to /mesh.py, run it, and call mesh_quality on
/mesh.msh when the artifacts are ready.

Use the OCC kernel to subtract every hole and rectangular void from the board
outline. Hole x/y values are centers; cutout, slot, and header x/y values are
also centers. Use a uniform {TARGET_SIZE_MM:g} mm target size, Frontal-Delaunay,
Blossom recombination, and smoothing. Write mesh.msh, mesh.vtk, and a mesh.png
preview without opening a GUI. Resolve all paths beside mesh.py.
"""


def _environment() -> dict[str, str]:
    return {
        "PATH": os.pathsep.join(
            (str(Path(sys.executable).parent), "/usr/local/bin", "/usr/bin", "/bin")
        ),
        "MPLBACKEND": "Agg",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _quality_tool(workspace: Path) -> BaseTool:
    @tool("mesh_quality", return_direct=True)
    def mesh_quality(mesh_file: str = "/mesh.msh") -> str:
        """Score the generated mesh and save quality.json."""
        mesh_path = (workspace / mesh_file.lstrip("/")).resolve()
        mesh_path.relative_to(workspace)
        board = json.loads((workspace / "board.json").read_text())
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "quadagent.quality",
                str(mesh_path),
                "--expected-area",
                str(quality.analytic_area(board)),
            ],
            cwd=workspace,
            env=_environment(),
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        report = json.loads(result.stdout)
        (workspace / "quality.json").write_text(json.dumps(report, indent=2) + "\n")
        return quality.summarize(report)

    return mesh_quality


def build_agent(
    workspace: Path, *, api_key: str, model: str = MODEL, base_url: str = BASE_URL
) -> Any:
    """Create the agent for one workspace."""
    workspace = workspace.resolve()
    backend = LocalShellBackend(
        root_dir=workspace,
        virtual_mode=True,
        timeout=120,
        env=_environment(),
        inherit_env=False,
    )
    chat_model = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.2,
        max_retries=2,
        use_responses_api=False,
    )
    register_harness_profile(
        f"openai:{model}",
        HarnessProfile(
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)
        ),
    )
    return create_deep_agent(
        model=chat_model,
        tools=[_quality_tool(workspace)],
        system_prompt=SYSTEM_PROMPT,
        backend=backend,
        subagents=[],
        middleware=[
            FilesystemMiddleware(
                backend=backend,
                tools=["read_file", "write_file", "execute"],
                max_execute_timeout=120,
            )
        ],
        name="quadagent",
    )


def run(board_name: str) -> Path:
    """Run one board and return its artifact directory."""
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("ZAI_API_KEY")
    if not api_key:
        raise RuntimeError("ZAI_API_KEY is not configured")
    board = BOARDS[board_name]
    board_id = json.loads(board.read_text())["board_id"]
    workspace = WORKSPACES / f"{board_id}-{datetime.now():%Y%m%d-%H%M%S-%f}"
    workspace.mkdir(parents=True)
    shutil.copyfile(board, workspace / "board.json")

    callbacks = []
    if os.environ.get("LANGSMITH_TRACING", "").lower() == "true" and os.environ.get(
        "LANGSMITH_API_KEY"
    ):
        callbacks.append(LangChainTracer())
    try:
        agent = build_agent(
            workspace,
            api_key=api_key,
            model=os.environ.get("QUADAGENT_MODEL", MODEL),
            base_url=os.environ.get("ZAI_BASE_URL", BASE_URL),
        )
        agent.invoke(
            {
                "messages": (
                    f"Mesh /board.json ({board_id}) at the configured target size."
                )
            },
            config={"callbacks": callbacks},
        )
    finally:
        wait_for_all_tracers()
    return workspace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("board", choices=sorted(BOARDS))
    args = parser.parse_args(argv)
    try:
        workspace = run(args.board)
    except Exception as error:  # noqa: BLE001 - command-line boundary
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(workspace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
