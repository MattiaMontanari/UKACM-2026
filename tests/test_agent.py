"""Offline end-to-end agent loop with a scripted (fake) chat model.

The fake model replays exactly the tool sequence the real LLM is prompted
to produce: plan -> write mesh.py -> execute -> quality -> report. No
network access happens anywhere in this test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quadagent.agent import DeepAgent
from quadagent.config import AgentConfig
from quadagent.tools import ToolRegistry

BOARD = {
    "board_id": "e2e-test",
    "outline": {"width": 12.0, "height": 12.0, "chamfers": []},
    "holes": [{"cx": 4.0, "cy": 4.0, "r": 1.2}, {"cx": 8.0, "cy": 8.0, "r": 1.2}],
    "cutouts": [{"cx": 8.0, "cy": 3.0, "w": 3.0, "h": 2.0}],
    "slots": [{"cx": 4.0, "cy": 9.0, "w": 1.0, "h": 3.0}],
    "header": {"cx": 6.0, "cy": 11.0, "w": 5.0, "h": 1.0},
}


class FakeModel:
    """Scripted chat model: returns queued tool calls, then a final answer."""

    def __init__(self, script: list[dict]) -> None:
        self.script = script
        self.cursor = 0
        self.last_usage: dict[str, int] = {}

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        entry = self.script[min(self.cursor, len(self.script) - 1)]
        self.cursor += 1
        self.last_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        return {"role": "assistant", "content": "", **entry}


def _mesh_script_source() -> str:
    from quadagent.recipes import recipe_source

    return recipe_source("level0_baseline")


def test_full_agent_loop(tmp_path: Path) -> None:
    (tmp_path / "board.json").write_text(json.dumps(BOARD))
    mesh_code = _mesh_script_source()
    model = FakeModel(
        [
            {
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "write_todos",
                            "arguments": json.dumps(
                                {
                                    "todos": [
                                        {"content": "write mesh.py", "status": "in_progress"},
                                        {"content": "run and check quality", "status": "pending"},
                                    ]
                                }
                            ),
                        },
                    }
                ]
            },
            {
                "tool_calls": [
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": json.dumps({"path": "mesh.py", "content": mesh_code}),
                        },
                    }
                ]
            },
            {
                "tool_calls": [
                    {
                        "id": "c3",
                        "type": "function",
                        "function": {"name": "execute_python", "arguments": json.dumps({"code": mesh_code})},
                    }
                ]
            },
            {
                "tool_calls": [
                    {
                        "id": "c4",
                        "type": "function",
                        "function": {
                            "name": "mesh_quality",
                            "arguments": json.dumps({"mesh_file": "mesh.msh"}),
                        },
                    }
                ]
            },
            {"content": "final report: baseline mesh delivered"},
        ]
    )
    registry = ToolRegistry(tmp_path, AgentConfig(sandbox_timeout_s=60))
    agent = DeepAgent(model, registry, max_steps=10)
    result = agent.run("mesh the board in board.json")

    assert result.stop_reason == "final_answer"
    assert result.n_model_calls == 5
    assert result.n_tool_calls == 4
    assert (tmp_path / "mesh.vtk").is_file()
    assert (tmp_path / "mesh.png").is_file()
    assert (tmp_path / "mesh.py").is_file()
    assert (tmp_path / "quality.json").is_file()
    assert (tmp_path / "_todos.md").is_file()
    transcript = (tmp_path / "transcript.jsonl").read_text().strip().splitlines()
    assert len(transcript) >= 10
    assert result.usage["total_tokens"] == 75

    quality_payload = json.loads((tmp_path / "quality.json").read_text())
    assert quality_payload["n_quads"] > 50
    assert quality_payload["quad_fraction"] >= 0.5


def test_model_error_salvages_run(tmp_path: Path) -> None:
    (tmp_path / "board.json").write_text(json.dumps(BOARD))

    class ExplodingModel:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: list[dict], tools: list[dict]) -> dict:
            self.calls += 1
            if self.calls >= 2:
                raise RuntimeError("simulated 429 quota exhaustion")
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "ls", "arguments": "{}"},
                    }
                ],
            }

    registry = ToolRegistry(tmp_path, AgentConfig())
    agent = DeepAgent(ExplodingModel(), registry, max_steps=10)
    result = agent.run("mesh the board")
    assert result.stop_reason == "model_error"
    assert "simulated 429" in result.final_text
    assert result.n_model_calls == 1


def test_path_jail(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path, AgentConfig())
    assert registry.dispatch("read_file", json.dumps({"path": "../../etc/passwd"})).startswith("error")
    assert registry.dispatch("write_file", json.dumps({"path": "/etc/pwned", "content": "x"})).startswith("error")
    assert registry.dispatch("read_file", json.dumps({"path": "missing.txt"})).startswith("error")


def test_reference_tool(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path, AgentConfig())
    full = registry.dispatch("read_reference", "{}")
    assert "playbook" in full.lower()
    topic = registry.dispatch("read_reference", json.dumps({"topic": "transfinite"}))
    assert "transfinite" in topic.lower()
