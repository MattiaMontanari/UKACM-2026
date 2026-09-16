"""Configuration from environment (with a tiny .env loader, no extra deps)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BASE_URL = "https://api.z.ai/api/coding/paas/v4"
DEFAULT_MODEL = "glm-5.3"


def load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from .env into os.environ without overriding."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class AgentConfig:
    """All knobs for one agent run."""

    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    max_steps: int = 30
    sandbox_timeout_s: int = 120
    sandbox_mode: str = "auto"
    max_tool_output_chars: int = 6000
    workspace_root: Path = field(default_factory=lambda: Path("workspace"))
    references_dir: Path | None = None

    @property
    def resolved_references_dir(self) -> Path:
        """References directory, defaulting to the repo checkout next to src/."""
        if self.references_dir is not None:
            return Path(self.references_dir)
        return Path(__file__).resolve().parents[2] / "references"


def load_config(**overrides: object) -> AgentConfig:
    """Build the configuration from env vars plus explicit overrides."""
    load_dotenv(Path(".env"))
    cfg = AgentConfig(
        api_key=os.environ.get("ZAI_API_KEY", ""),
        base_url=os.environ.get("ZAI_BASE_URL", DEFAULT_BASE_URL),
        model=os.environ.get("QUADAGENT_MODEL", DEFAULT_MODEL),
        sandbox_mode=os.environ.get("QUADAGENT_SANDBOX", "auto"),
        references_dir=(
            Path(os.environ["QUADAGENT_REFERENCES"])
            if os.environ.get("QUADAGENT_REFERENCES")
            else None
        ),
    )
    for key, value in overrides.items():
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg
