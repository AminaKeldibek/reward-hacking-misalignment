"""Load the `knownliebench:` block of a run config (misalignment-evals/configs/eval_run.yaml)."""
from pathlib import Path
from typing import Literal, Optional, get_args

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from misalignment_evals.agent_loop import AgentLoop, load_agent_loop
from misalignment_evals.run_config import Generation, load_suite_block

Domain = Literal[
    "refund", "airline", "deposit", "recall", "subscription", "billing", "insurance", "debt"]
Condition = Literal["none", "lie_incentive", "instructed_lie"]
TrustLevel = Literal["high", "neutral", "low"]

DOMAINS: tuple[str, ...] = get_args(Domain)
CONDITIONS: tuple[str, ...] = get_args(Condition)
TRUST_LEVELS: tuple[str, ...] = get_args(TrustLevel)
CASES_PER_DOMAIN = 14   # 8 owed cases, then 6 calibration (nothing owed) cases, in that order


class Role(BaseModel):
    """A fixed measurement model (the simulated customer or the judge). Always sampled at temp 0."""

    model_config = ConfigDict(extra="forbid")
    model: str
    max_tokens: StrictInt = Field(ge=1)
    reasoning_effort: Optional[Literal["none", "minimal", "low", "medium", "high"]]


class Execution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_tasks: StrictInt = Field(ge=1)   # up to 3 condition tasks + the knowledge task
    max_samples: StrictInt = Field(ge=1)
    max_connections: StrictInt = Field(ge=1)
    time_limit: Optional[int]
    fail_on_error: Optional[float]


class KnownLieBenchConfig(BaseModel):
    """The whole `knownliebench:` block. Every key is required (no code defaults). The paper's
    protocol: customer gpt-4o-mini-2024-07-18 (600 tokens), judge gpt-5.1 (900 tokens, reasoning
    none), all 8 domains x 3 conditions x 3 trust levels, 14 cases, 1 trajectory per cell."""

    model_config = ConfigDict(extra="forbid")

    # model / model_base_url / reasoning_tag / generation are filled from `evaluated_model:` by
    # load_knownliebench_config.
    model: str
    model_base_url: Optional[str]
    reasoning_tag: str
    customer: Role
    judge: Role
    generation: Generation
    execution: Execution
    domains: list[Domain]
    conditions: list[Condition]
    trust_levels: list[TrustLevel]
    # The first `cases` cases of each domain go to the knowledge check; only the known ones are played.
    cases: StrictInt = Field(ge=1, le=CASES_PER_DOMAIN)
    epochs: StrictInt = Field(ge=1)
    # Not in the paper (which gates each round once at temperature 0): before any trajectory runs,
    # each case's knowledge question is sampled `knowledge_samples` times at the agent's
    # temperature, and only cases answered right in >= `knowledge_threshold` of them are played, the
    # same list in every condition.
    knowledge_samples: StrictInt = Field(ge=1)
    knowledge_threshold: float = Field(gt=0, le=1)
    # Not in the paper: an agent output that isn't one usable JSON action is never sent to the
    # customer; the parse error goes back to the agent, which may retry this many times per turn.
    # If every retry fails, the round is invalid (end_reason "parse_error"). Required: no default.
    max_parse_retries: StrictInt = Field(ge=0)
    # The top-level `agent_loop:` group. Only `max_repeated_outputs` applies here: the agent's turn
    # is a JSON action with its own "thought", not a tool call.
    agent_loop: AgentLoop

    @model_validator(mode="after")
    def _lists_nonempty(self):
        for name in ("domains", "conditions", "trust_levels"):
            if not getattr(self, name):
                raise ValueError(f"{name} is empty — nothing would run")
        return self


def load_knownliebench_config(path: "str | Path") -> KnownLieBenchConfig:
    """Parse the config into a KnownLieBenchConfig. The model under test (model / model_base_url /
    reasoning_tag / generation) comes from the shared top-level `evaluated_model:` group, the agent
    loop from `agent_loop:`; the benchmark's own settings (customer, judge, budget) from
    `knownliebench:`."""
    return load_suite_block(
        path, "knownliebench", KnownLieBenchConfig,
        ("model", "model_base_url", "reasoning_tag", "generation"),
        hint=(f"Valid domains: {', '.join(DOMAINS)}; conditions: {', '.join(CONDITIONS)}; "
              f"trust levels: {', '.join(TRUST_LEVELS)}"),
        shared_groups={"agent_loop": load_agent_loop(path)},
    )
