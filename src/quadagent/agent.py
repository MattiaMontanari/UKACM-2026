"""The agent loop: a single-agent ReAct cycle with tools and a step budget.

This is deliberately framework-free. It mirrors what the `deepagents`
library gives you minus subagents: system prompt + planning + tool loop +
persistent workspace. Every message is appended to transcript.jsonl in the
session workspace so lectures can replay exactly what the agent did.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from quadagent.prompts import SYSTEM_PROMPT
from quadagent.tools import ToolRegistry

WRAP_UP_WARNING = (
    "You have 2 steps left. Stop exploring: make sure mesh.vtk, mesh.png "
    "and mesh.py exist and are your best attempt, then give the final report."
)


@dataclass
class AgentResult:
    """Outcome of one full agent run."""

    final_text: str = ""
    n_model_calls: int = 0
    n_tool_calls: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    tool_trace: list[str] = field(default_factory=list)
    stop_reason: str = "final_answer"


class DeepAgent:
    """One agent session bound to a workspace, tools and a chat model."""

    def __init__(
        self,
        model: Any,
        registry: ToolRegistry,
        *,
        max_steps: int = 30,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.model = model
        self.registry = registry
        self.max_steps = max_steps
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.transcript_path = registry.workspace / "transcript.jsonl"
        self._log = log or (lambda _msg: None)

    def run(self, task: str) -> AgentResult:
        """Execute the task to completion (or budget exhaustion)."""
        result = AgentResult()
        self.messages.append({"role": "user", "content": task})
        self._record({"role": "user", "content": task})

        for step in range(1, self.max_steps + 1):
            if step == self.max_steps - 1:
                self.messages.append({"role": "user", "content": WRAP_UP_WARNING})
                self._record({"role": "user", "content": WRAP_UP_WARNING, "meta": "wrap-up warning"})
            self._log(f"step {step}/{self.max_steps}: thinking...")
            started = time.monotonic()
            try:
                message = self.model.complete(self.messages, self.registry.specs())
            except Exception as exc:  # noqa: BLE001 - salvage partial runs
                result.stop_reason = "model_error"
                result.final_text = (
                    f"model call failed at step {step}: "
                    f"{type(exc).__name__}: {str(exc)[:300]}\n"
                    "Partial run salvaged: check the workspace artifacts "
                    "(best mesh so far, mesh.py, transcript.jsonl) and "
                    "quality.json for the last measured state."
                )
                self._log(f"model error at step {step}: {exc}")
                (self.registry.workspace / "usage.json").write_text(
                    json.dumps(result.usage, indent=2) + "\n"
                )
                return result
            result.n_model_calls += 1
            usage = getattr(self.model, "last_usage", None)
            if usage:
                for key, value in usage.items():
                    result.usage[key] = result.usage.get(key, 0) + value
                self._record({"role": "usage", "content": json.dumps(usage)})
            self.messages.append(message)
            self._record(message)
            elapsed = time.monotonic() - started

            calls = message.get("tool_calls") or []
            if not calls:
                result.final_text = message.get("content") or ""
                self._log(f"final answer after {step} steps ({elapsed:.1f}s)")
                (self.registry.workspace / "usage.json").write_text(
                    json.dumps(result.usage, indent=2) + "\n"
                )
                return result

            for call in calls:
                name = call["function"]["name"]
                arguments = call["function"].get("arguments", "{}")
                self._log(f"  tool {name} ...")
                output = self.registry.dispatch(name, arguments)
                result.n_tool_calls += 1
                status = "error" if output.startswith("error") else "ok"
                result.tool_trace.append(f"{name}:{status}")
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": output,
                }
                self.messages.append(tool_message)
                self._record(tool_message)

        result.stop_reason = "step_budget_exhausted"
        result.final_text = (
            "step budget exhausted before a final answer; "
            "check the workspace artifacts and transcript for the last state"
        )
        (self.registry.workspace / "usage.json").write_text(
            json.dumps(result.usage, indent=2) + "\n"
        )
        return result

    def _record(self, message: dict[str, Any]) -> None:
        entry = {
            "ts": time.time(),
            **message,
        }
        with self.transcript_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
