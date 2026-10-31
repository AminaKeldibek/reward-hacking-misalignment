"""Load the `budget_misrepresentation:` block of a run config (misalignment-evals/configs/eval_run.yaml).

Every key is required (no code defaults). The model under test (model, model_base_url, reasoning_tag,
generation) comes from `evaluated_model:`, the agent loop from `agent_loop:`. Only
`agent_loop.max_repeated_outputs` applies: the system prompt keeps the paper's own reasoning line
(private <reasoning_tag> reasoning "NOT shown to the user"), so `agent_loop.reasoning` is not used.
"""
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from misalignment_evals.agent_loop import AgentLoop, load_agent_loop
from misalignment_evals.budget_misrepresentation.prompts import SNIPPET_NAMES
from misalignment_evals.run_config import Generation, load_suite_block
from misalignment_evals.scorers.base import ModelRole, ScorerConfig

BLOCK = "budget_misrepresentation"
Level = Literal["high", "low"]


class Environment(BaseModel):
    """The paper's environmental factors (Table 5). Its baseline: stakes high, oversight low,
    outcome_influence high."""

    model_config = ConfigDict(extra="forbid")
    stakes: Level
    oversight: Level
    outcome_influence: Level


class Execution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_samples: StrictInt = Field(ge=1)
    max_connections: StrictInt = Field(ge=1)
    message_limit: StrictInt = Field(ge=2)
    time_limit: Optional[int]
    fail_on_error: Optional[float]


class BudgetMisrepresentationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    model_base_url: Optional[str]
    reasoning_tag: str
    generation: Generation
    agent_loop: AgentLoop
    judge: ModelRole
    eval_awareness_judge: Optional[ScorerConfig]
    execution: Execution
    environment: Environment
    tools: Literal[0, 3, 4, 7]
    snippets: list[str]
    epochs: StrictInt = Field(ge=1)

    @field_validator("snippets")
    @classmethod
    def _known_snippets(cls, snippets: list[str]) -> list[str]:
        if not snippets:
            raise ValueError("empty — nothing would run")
        unknown = [s for s in snippets if s not in SNIPPET_NAMES]
        if unknown:
            raise ValueError(f"unknown {unknown}")
        if len(set(snippets)) != len(snippets):
            raise ValueError("a snippet is listed twice")
        return snippets


def load_budget_misrepresentation_config(path: "str | Path") -> BudgetMisrepresentationConfig:
    return load_suite_block(
        path, BLOCK, BudgetMisrepresentationConfig,
        ("model", "model_base_url", "reasoning_tag", "generation"),
        hint=f"Valid snippets: {', '.join(SNIPPET_NAMES)}; tools: 0, 3, 4, 7",
        shared_groups={"agent_loop": load_agent_loop(path)},
    )
