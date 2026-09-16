"""Agent tools: planning, virtual filesystem, sandboxed execution, quality.

Every tool takes JSON arguments and returns a plain string (errors included
as `error: ...` text) so the agent loop never sees host exceptions. File
tools are jailed to the session workspace.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from quadagent import quality
from quadagent.config import AgentConfig
from quadagent.sandbox import run_python


@dataclass(frozen=True)
class Tool:
    """One tool: OpenAI schema plus its implementation."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[[dict[str, Any]], str]


def _object(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


class ToolRegistry:
    """Builds and dispatches the tool set for one session workspace."""

    def __init__(self, workspace: Path, cfg: AgentConfig) -> None:
        self.workspace = Path(workspace).resolve()
        self.cfg = cfg
        self._call_counter = 0
        self.tools: dict[str, Tool] = {tool.name: tool for tool in self._build()}

    def specs(self) -> list[dict[str, Any]]:
        """OpenAI `tools=` payload."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self.tools.values()
        ]

    def dispatch(self, name: str, arguments: str | dict) -> str:
        """Run one tool call; always answers with a string."""
        tool = self.tools.get(name)
        if tool is None:
            return f"error: unknown tool {name!r}; available: {sorted(self.tools)}"
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
            return tool.fn(args)
        except Exception as exc:  # noqa: BLE001 - tools must not raise
            return f"error: {type(exc).__name__}: {exc}"

    def _safe(self, rel: str, *, must_exist: bool = False) -> Path:
        """Resolve a workspace-relative path, refusing escapes."""
        if not isinstance(rel, str) or not rel:
            raise ValueError("path must be a non-empty string")
        candidate = Path(rel)
        if candidate.is_absolute():
            raise ValueError("absolute paths are not allowed; use workspace-relative paths")
        resolved = (self.workspace / candidate).resolve()
        if resolved != self.workspace and self.workspace not in resolved.parents:
            raise ValueError(f"path {rel!r} escapes the workspace")
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"{rel!r} does not exist in the workspace")
        return resolved

    def _truncate(self, text: str) -> str:
        limit = self.cfg.max_tool_output_chars
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n... [truncated, {len(text)} chars total]"

    def _board_expected_area(self) -> float | None:
        board = self._safe("board.json", must_exist=True)
        return quality.analytic_area(json.loads(board.read_text()))

    def _build(self) -> list[Tool]:
        return [
            self._tool_write_todos(),
            self._tool_ls(),
            self._tool_read_file(),
            self._tool_write_file(),
            self._tool_execute_python(),
            self._tool_mesh_quality(),
            self._tool_read_reference(),
        ]

    def _tool_write_todos(self) -> Tool:
        def fn(args: dict[str, Any]) -> str:
            todos = args.get("todos", [])
            if not isinstance(todos, list):
                raise ValueError("todos must be an array")
            markers = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
            lines = ["# Plan", ""]
            for item in todos:
                status = item.get("status", "pending")
                if status not in markers:
                    raise ValueError(f"invalid todo status {status!r}")
                lines.append(f"{markers[status]} {item.get('content', '')}")
            (self.workspace / "_todos.md").write_text("\n".join(lines) + "\n")
            return "\n".join(lines)

        return Tool(
            name="write_todos",
            description=(
                "Update the task plan. Pass the full todo list each call; "
                "statuses: pending | in_progress | completed. Start complex "
                "work by writing the plan, keep exactly one in_progress item."
            ),
            parameters=_object(
                {
                    "todos": {
                        "type": "array",
                        "items": _object(
                            {
                                "content": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                },
                            },
                            ["content", "status"],
                        ),
                    }
                },
                ["todos"],
            ),
            fn=fn,
        )

    def _tool_ls(self) -> Tool:
        def fn(args: dict[str, Any]) -> str:
            base = self._safe(args.get("path", "."), must_exist=True)
            if base.is_file():
                stat = base.stat()
                return f"{args.get('path', '.')} (file, {stat.st_size} bytes)"
            entries = sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            if not entries:
                return "(empty directory)"
            lines = []
            for entry in entries:
                if entry.is_dir():
                    lines.append(f"{entry.name}/")
                else:
                    lines.append(f"{entry.name} ({entry.stat().st_size} bytes)")
            return self._truncate("\n".join(lines))

        return Tool(
            name="ls",
            description="List a workspace directory (or describe one file).",
            parameters=_object({"path": {"type": "string"}}, []),
            fn=fn,
        )

    def _tool_read_file(self) -> Tool:
        def fn(args: dict[str, Any]) -> str:
            path = self._safe(args["path"], must_exist=True)
            if path.is_dir():
                raise ValueError(f"{args['path']!r} is a directory")
            return self._truncate(path.read_text(errors="replace"))

        return Tool(
            name="read_file",
            description=(
                "Read a text file from the workspace (board.json, your "
                "scripts, the gmsh output logs)."
            ),
            parameters=_object({"path": {"type": "string"}}, ["path"]),
            fn=fn,
        )

    def _tool_write_file(self) -> Tool:
        def fn(args: dict[str, Any]) -> str:
            path = self._safe(args["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            content = args.get("content", "")
            if not isinstance(content, str):
                raise ValueError("content must be a string")
            path.write_text(content)
            return f"wrote {args['path']} ({len(content)} chars)"

        return Tool(
            name="write_file",
            description=(
                "Write a text file in the workspace (e.g. mesh.py). "
                "Overwrites existing files."
            ),
            parameters=_object(
                {"path": {"type": "string"}, "content": {"type": "string"}},
                ["path", "content"],
            ),
            fn=fn,
        )

    def _tool_execute_python(self) -> Tool:
        def fn(args: dict[str, Any]) -> str:
            code = args.get("code")
            if not isinstance(code, str) or not code.strip():
                raise ValueError("code must be a non-empty string")
            self._call_counter += 1
            result, script = run_python(
                code,
                cwd=self.workspace,
                timeout_s=self.cfg.sandbox_timeout_s,
                mode=self.cfg.sandbox_mode,
                call_id=self._call_counter,
            )
            if result.timed_out:
                status = f"TIMEOUT after {self.cfg.sandbox_timeout_s}s"
            elif result.exit_code == 0:
                status = "OK"
            else:
                status = f"EXIT {result.exit_code}"
            parts = [
                f"status: {status}",
                f"script: {script.relative_to(self.workspace)}",
                f"duration_s: {result.duration_s:.2f}",
            ]
            if result.stdout.strip():
                parts.append("stdout:\n" + self._truncate(result.stdout))
            if result.stderr.strip():
                parts.append("stderr:\n" + self._truncate(result.stderr))
            return "\n".join(parts)

        return Tool(
            name="execute_python",
            description=(
                "Run Python code in a sandboxed subprocess (no network, "
                "workspace as cwd, headless). `import gmsh`, numpy and "
                "matplotlib are available; board.json is already in cwd. "
                "Write mesh outputs (mesh.msh, mesh.vtk, mesh.png) into "
                "the current directory."
            ),
            parameters=_object({"code": {"type": "string"}}, ["code"]),
            fn=fn,
        )

    def _tool_mesh_quality(self) -> Tool:
        def fn(args: dict[str, Any]) -> str:
            mesh_path = self._safe(args.get("mesh_file", "mesh.msh"), must_exist=True)
            expected = args.get("expected_area_mm2")
            if expected is None and self._safe("board.json").is_file():
                expected = self._board_expected_area()
            report = quality.analyze(
                mesh_path,
                expected_area_mm2=expected,
                wave_speed_m_s=float(args.get("wave_speed_m_s", quality.DEFAULT_WAVE_SPEED_M_S)),
            )
            (self.workspace / "quality.json").write_text(
                quality.summarize_json(report) + "\n"
            )
            return quality.summarize(report)

        return Tool(
            name="mesh_quality",
            description=(
                "Compute quality metrics of a mesh file (.msh or .vtk): "
                "quad fraction, scaled Jacobian, aspect ratio, min angle, "
                "edge lengths, critical time step, area check against "
                "board.json; plus the 3 WORST elements with their (x, y) "
                "positions so you know where to fix, per-surface coverage, "
                "and odd-boundary-parity warnings (a surface with odd "
                "parity cannot fully recombine — it will leave triangles). "
                "Verdicts use the playbook gates."
            ),
            parameters=_object(
                {
                    "mesh_file": {"type": "string"},
                    "expected_area_mm2": {"type": "number"},
                    "wave_speed_m_s": {"type": "number"},
                },
                ["mesh_file"],
            ),
            fn=fn,
        )

    def _tool_read_reference(self) -> Tool:
        def fn(args: dict[str, Any]) -> str:
            refs_dir = self.cfg.resolved_references_dir
            allowed = ("playbook.md", "gmsh_quads.md")
            files = [refs_dir / name for name in allowed if (refs_dir / name).is_file()]
            texts = []
            for path in files:
                if path.is_file():
                    texts.append(f"===== {path.name} =====\n{path.read_text()}")
            if not texts:
                return "error: no references found"
            joined = "\n\n".join(texts)
            topic = args.get("topic")
            if topic:
                keywords = [k.lower() for k in str(topic).replace(",", " ").split() if k]
                sections: list[str] = []
                for block in joined.split("\n## "):
                    lowered = block.lower()
                    if any(k in lowered for k in keywords):
                        sections.append(block if block.startswith("## ") else "## " + block)
                if sections:
                    return self._truncate("\n\n".join(sections))
                return (
                    f"no reference section matched {topic!r}; call without "
                    "topic to read the full cheat sheet and playbook"
                )
            return self._truncate(joined)

        return Tool(
            name="read_reference",
            description=(
                "Read the bundled quad-meshing playbook and Gmsh API cheat "
                "sheet. Pass topic (e.g. 'recombine', 'transfinite', "
                "'fields') to pull only matching sections."
            ),
            parameters=_object({"topic": {"type": "string"}}, []),
            fn=fn,
        )
