"""The top-level `agent_loop:` group of a run config: one setting for every multi-turn eval."""
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

from misalignment_evals.run_config import format_errors

GROUP = "agent_loop"


class AgentLoop(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_repeated_outputs: StrictInt = Field(ge=2)
    reasoning: Literal["rl_prompt", "none"]


def load_agent_loop(path: "str | Path") -> AgentLoop:
    """The validated `agent_loop:` group. Every key is required."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"--config {p} not found")
    loaded = yaml.safe_load(p.read_text()) or {}
    if GROUP not in loaded:
        raise SystemExit(f"config {p} has no top-level `{GROUP}:` group "
                         f"(keys: {', '.join(AgentLoop.model_fields)})")
    try:
        return AgentLoop.model_validate(loaded[GROUP] or {})
    except ValidationError as e:
        raise SystemExit(f"config `{GROUP}:` is invalid — {format_errors(e, GROUP)}") from None
