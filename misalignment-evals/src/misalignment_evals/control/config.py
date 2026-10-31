"""Load the `control_evals:` block of a run config (misalignment-evals/configs/control_run.yaml)."""
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from misalignment_evals.agent_loop import AgentLoop, load_agent_loop
from misalignment_evals.control.settings import ARM_NAMES, EVAL_NAMES
from misalignment_evals.run_config import Generation, load_suite_block


class CaseSelection(BaseModel):
    """Run the non-baseline arms only on cases the model solved in the baseline arm."""

    model_config = ConfigDict(extra="forbid")
    min_solved_epochs: StrictInt = Field(ge=1)   # baseline epochs a case must pass to be selected


class EvalBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    arms: list[Literal["baseline", "instructed", "propensity"]]
    samples: StrictInt = Field(ge=1)   # distinct cases drawn (with case_selection: cases screened in baseline)
    epochs: StrictInt = Field(ge=1)    # completions per case
    case_selection: Optional[CaseSelection]   # required key; null = every arm runs on the first `samples` cases

    @model_validator(mode="after")
    def _arms_nonempty(self):
        if not self.arms:
            raise ValueError("arms is empty — nothing would run")
        if self.case_selection is not None:
            if "baseline" not in self.arms:
                raise ValueError("case_selection selects from the baseline arm, so arms must include baseline")
            if self.case_selection.min_solved_epochs > self.epochs:
                raise ValueError(f"case_selection.min_solved_epochs ({self.case_selection.min_solved_epochs}) "
                                 f"exceeds epochs ({self.epochs})")
        return self


class AppsBudget(EvalBudget):
    """apps_backdoor also chooses its feedback (control/run_tests.py)."""

    feedback: Literal["none", "visible_tests"]   # visible_tests: run_tests on the statement's examples
    max_test_runs: Optional[StrictInt] = Field(ge=1)   # run_tests calls per episode; null with none

    @model_validator(mode="after")
    def _budget_matches_feedback(self):
        if (self.feedback == "visible_tests") != (self.max_test_runs is not None):
            raise ValueError("max_test_runs is set with feedback: visible_tests, and null with none")
        return self


class Execution(BaseModel):
    """Run mechanics. Every key is required; a null limit means none (control-arena's default)."""

    model_config = ConfigDict(extra="forbid")
    max_tasks: StrictInt = Field(ge=1)
    max_samples: StrictInt = Field(ge=1)
    max_connections: StrictInt = Field(ge=1)
    max_sandboxes: StrictInt = Field(ge=1)   # concurrent Docker sandboxes across ALL arms; bash needs 2 networks each
    token_limit: Optional[int]
    message_limit: Optional[int]           # agentic step cap (bash)
    time_limit: Optional[int]
    fail_on_error: Optional[float]
    # control-arena's recoverable tool-output excerpting: adds read_tool_output / grep_tool_output
    tool_output_recovery: bool


class ControlConfig(BaseModel):
    """The whole `control_evals:` block. Every key is required (no code defaults)."""

    model_config = ConfigDict(extra="forbid")

    model: str                              # untrusted model organism, e.g. openai-api/vllm/ckpt50
    model_base_url: Optional[str]           # -> VLLM_BASE_URL (vLLM served); null for API providers
    reasoning_tag: str                      # from evaluated_model: the tag the model was RL-trained on
    agent_loop: AgentLoop                   # the top-level `agent_loop:` group
    monitor_model: str                      # trusted monitor (the judge), e.g. openrouter/...
    apps_sandbox: Literal["docker", "podman", "local"]
    cases_file: Optional[str]               # selected-cases manifest from select_cases (task-id whitelist)
    generation: Generation
    execution: Execution
    evals: dict[Literal["apps_backdoor", "bash_exfil"], "AppsBudget | EvalBudget"]

    @model_validator(mode="after")
    def _evals_nonempty(self):
        if not self.evals:
            raise ValueError("no evals listed — nothing would run")
        for name, budget in self.evals.items():
            if (name == "apps_backdoor") != isinstance(budget, AppsBudget):
                raise ValueError("feedback and max_test_runs are required for apps_backdoor and "
                                 f"apply to it only ({name})")
        if self.cases_file and any(b.case_selection is not None for b in self.evals.values()):
            raise ValueError("cases_file (a frozen case list) and case_selection (select in this run) "
                             "are mutually exclusive — set one")
        return self


def load_control_config(path: "str | Path") -> ControlConfig:
    """Parse the config into a ControlConfig. The model under test (model / model_base_url /
    reasoning_tag / generation) comes from the shared top-level `evaluated_model:` group, the agent
    loop from `agent_loop:`; control-specific settings (monitor, arms, sandbox, cases) from
    `control_evals:`."""
    return load_suite_block(
        path, "control_evals", ControlConfig,
        ("model", "model_base_url", "reasoning_tag", "generation"),
        hint=f"Valid evals: {', '.join(EVAL_NAMES)}; valid arms: {', '.join(ARM_NAMES)}",
        shared_groups={"agent_loop": load_agent_loop(path)},
    )


def evals_arms(cfg: ControlConfig) -> dict[str, list[str]]:
    """{eval_name: [arm, ...]} for build_tasks — the config's `evals:` block is the include list."""
    return {name: budget.arms for name, budget in cfg.evals.items()}
