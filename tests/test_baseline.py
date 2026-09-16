"""Deterministic contract tests for the lecture baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage
from langgraph.graph import END
from langgraph.types import Command

import baseline

EXPECTED_BOARDS = {
    "holes": "train-holes-00",
    "rects": "train-rects-00",
    "slots": "train-slots-00",
    "dense": "train-dense-00",
}


def test_board_aliases_resolve_to_the_four_active_domains() -> None:
    assert set(baseline.BOARDS) == set(EXPECTED_BOARDS)
    for alias, board_id in EXPECTED_BOARDS.items():
        path = baseline.resolve_board(alias)
        assert path.name == f"{board_id}.json"
        assert json.loads(path.read_text())["board_id"] == board_id


def test_unknown_board_alias_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown board"):
        baseline.resolve_board("unknown")


def test_agent_exposes_only_the_part_1_tool_surface(tmp_path: Path) -> None:
    (tmp_path / "board.json").write_text(
        json.dumps(
            {
                "board_id": "contract-test",
                "outline": {"width": 10.0, "height": 10.0, "chamfers": []},
                "holes": [],
                "cutouts": [],
                "slots": [],
                "header": None,
            }
        )
    )

    agent = baseline.build_agent(
        tmp_path,
        api_key="not-used",
        model_name="glm-5.3",
        base_url="https://example.invalid/v1",
    )
    tools = agent.nodes["tools"].bound._tools_by_name

    assert set(tools) == {
        "execute",
        "mesh_quality",
        "read_file",
        "write_file",
        "write_todos",
    }
    assert tools["mesh_quality"].return_direct is True


def test_baseline_prompt_forbids_quality_driven_refinement() -> None:
    prompt = " ".join(baseline.SYSTEM_PROMPT.split())
    assert "exactly once" in prompt
    assert "Do not modify" in prompt
    assert "O-grid" in prompt
    assert "never `cd /`" in prompt
    assert "lower-left corner (0, 0)" in prompt
    assert "relative to the directory containing mesh.py" in prompt
    assert "node_tags, coords, _ = gmsh.model.mesh.getNodes()" in prompt
    assert "only tool call in that model response" in prompt


def test_mesh_quality_middleware_jumps_directly_to_graph_end() -> None:
    message = ToolMessage(content="scored", tool_call_id="score-1", name="mesh_quality")
    request = SimpleNamespace(tool_call={"name": "mesh_quality", "id": "score-1"})

    result = baseline._terminal_mesh_quality.wrap_tool_call(request, lambda _request: message)

    assert isinstance(result, Command)
    assert result.goto == END
    assert result.update == {"messages": [message]}


def test_shell_environment_uses_the_active_uv_environment() -> None:
    env = baseline._shell_environment()

    assert env["PATH"].split(":")[0] == str(Path(sys.executable).parent)
    assert not any("API_KEY" in name for name in env)
