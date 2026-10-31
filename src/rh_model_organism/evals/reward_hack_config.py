"""Load the `reward_hacking:` block of a run config (misalignment-evals/configs/eval_run.yaml).

Every key is required (no code defaults). The model under test (model, model_base_url,
reasoning_tag, generation) comes from `evaluated_model:`, the agent loop from `agent_loop:`. The
`evals:` block is the include list: an eval runs if and only if it appears there.

    reward_hacking:
      max_connections: 8
      max_sandboxes: 8
      fail_on_error: 0.1
      agent: {truncation: auto, message_limit: 100, tool_timeout: 180, tool_output_chars: 3000,
              read_file_chars: 25000}
      evals:
        impossible_lcb: {samples: 50, epochs: 2, split: conflicting, test_instruction: dont_modify,
                         hack_hints: none, attempts: 3}
"""
from pathlib import Path
from typing import Literal, Optional

from misalignment_evals.agent_loop import AgentLoop, load_agent_loop
from misalignment_evals.run_config import Generation, load_suite_block
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

EVAL_NAMES: tuple[str, ...] = ("impossible_lcb", "evilgenie")
HackHints = Literal["none", "names", "rl_text"]


class Agent(BaseModel):
    """The react() agent both evals run on."""

    model_config = ConfigDict(extra="forbid")
    truncation: Literal["auto", "disabled"]     # react(): trim old messages on overflow, or end
    message_limit: StrictInt = Field(ge=1)
    tool_timeout: StrictInt = Field(ge=1)       # seconds per safe_bash / safe_python call
    tool_output_chars: StrictInt = Field(ge=1)  # command output beyond this is cut in the middle
    read_file_chars: StrictInt = Field(ge=1)    # read_file returns up to this much of a file


class Budget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    samples: StrictInt = Field(ge=1)
    epochs: StrictInt = Field(ge=1)
    hack_hints: HackHints


class ImpossibleLcb(Budget):
    split: Literal["original", "oneoff", "conflicting"]
    test_instruction: Literal["dont_modify", "none"]
    attempts: StrictInt = Field(ge=1)           # submissions; all tests are visible, so feedback is safe


class EvilGenie(Budget):
    difficulty: Literal["easy", "medium", "hard"]
    dataset_source: Literal["livecodebench", "apps"]
    seed: StrictInt
    judge_model: Optional[str]                  # null: no LLM judge, only the deterministic signals


class Evals(BaseModel):
    """The include list. An eval left out does not run."""

    model_config = ConfigDict(extra="forbid")
    impossible_lcb: Optional[ImpossibleLcb] = None
    evilgenie: Optional[EvilGenie] = None

    @model_validator(mode="after")
    def _nonempty(self):
        if self.impossible_lcb is None and self.evilgenie is None:
            raise ValueError("no evals listed — nothing would run")
        return self


class RewardHackingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    model_base_url: Optional[str]
    reasoning_tag: str
    generation: Generation
    agent_loop: AgentLoop
    max_connections: StrictInt = Field(ge=1)
    max_sandboxes: StrictInt = Field(ge=1)      # concurrent Docker sandboxes (8 fits the 10-CPU Mac)
    fail_on_error: Optional[float]
    agent: Agent
    evals: Evals


def load_reward_hack_config(path: "str | Path") -> RewardHackingConfig:
    return load_suite_block(
        path, "reward_hacking", RewardHackingConfig,
        ("model", "model_base_url", "reasoning_tag", "generation"),
        hint=f"Valid evals: {', '.join(EVAL_NAMES)}",
        shared_groups={"agent_loop": load_agent_loop(path)},
    )


def eval_settings(cfg: RewardHackingConfig, name: str) -> "ImpossibleLcb | EvilGenie":
    """One eval's settings, or a loud error naming what the config does configure."""
    settings = getattr(cfg.evals, name, None)
    if settings is None:
        configured = [n for n in EVAL_NAMES if getattr(cfg.evals, n) is not None]
        raise SystemExit(f"--eval {name} is not in the config's `reward_hacking: evals:` block. "
                         f"Configured: {', '.join(configured)}")
    return settings
