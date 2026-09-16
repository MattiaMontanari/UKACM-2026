"""Thin LLM client for the GLM coding endpoint (OpenAI-compatible).

The agent loop only needs `complete(messages, tools) -> message dict`, so
tests can inject any object with the same method (see FakeLLM in tests).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from quadagent.config import AgentConfig


class ChatModel(Protocol):
    """Minimal interface the agent loop depends on."""

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        """Return one assistant message as a plain dict."""
        ...


class GlmClient:
    """OpenAI-compatible chat client pointed at the z.ai coding endpoint."""

    def __init__(self, cfg: AgentConfig, client: Any | None = None) -> None:
        if client is None:
            if not cfg.api_key:
                msg = "ZAI_API_KEY is not set (put it in .env, see .env.example)"
                raise RuntimeError(msg)
            from openai import OpenAI

            client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        self._client = client
        self._model = cfg.model
        self.last_usage: dict[str, int] = {}

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        """One chat completion; returns the assistant message as a dict."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
        )
        if response.usage is not None:
            self.last_usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }
        message = response.choices[0].message
        return message.model_dump(exclude_none=True)
