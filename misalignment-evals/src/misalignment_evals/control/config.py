"""Load the `control_evals:` block of a run config (misalignment-evals/configs/control_run.yaml)."""
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from misalignment_evals.control.settings import ARM_NAMES, EVAL_NAMES
from misalignment_evals.run_config import Generation, load_suite_block


class EvalBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    arms: list[Literal["baseline", "instructed", "propensity"]] = list(ARM_NAMES)
    samples: StrictInt = Field(ge=1)   # distinct cases drawn from the eval's dataset
    epochs: StrictInt = Field(ge=1)    # completions per case

    @model_validator(mode="after")
    def _arms_nonempty(self):
        if not self.arms:
            raise ValueError("arms is empty — nothing would run")
        return self


class Execution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_tasks: int = 4
    max_samples: int = 100
    max_connections: int = 16
    token_limit: Optional[int] = None
    message_limit: Optional[int] = None    # agentic step cap (bash); None = control-arena default
    time_limit: Optional[int] = None
    fail_on_error: Optional[float] = 0.1


class ControlConfig(BaseModel):
    """The whole `control_evals:` block."""

    model_config = ConfigDict(extra="forbid")

    # model / model_base_url / monitor_model may come from the config OR the CLI (the per-checkpoint
    # wrapper injects the served model). They are resolved + required at run time, not parse time.
    model: Optional[str] = None             # untrusted model organism, e.g. openai-api/vllm/ckpt50
    model_base_url: Optional[str] = None    # -> VLLM_BASE_URL (vLLM served); None for API providers
    monitor_model: Optional[str] = None     # trusted monitor (the judge), e.g. openrouter/...
    apps_sandbox: Literal["docker", "podman", "local"] = "docker"
    cases_file: Optional[str] = None        # selected-cases manifest from select_cases (task-id whitelist)
    generation: Generation = Generation()
    execution: Execution = Execution()
    evals: dict[Literal["apps_backdoor", "bash_exfil"], EvalBudget]

    @model_validator(mode="after")
    def _evals_nonempty(self):
        if not self.evals:
            raise ValueError("no evals listed — nothing would run")
        return self


def load_control_config(path: "str | Path") -> ControlConfig:
    """Parse the config into a ControlConfig. The model under test (model / model_base_url /
    generation) comes from the shared top-level `evaluated_model:` group; control-specific settings
    (monitor, arms, sandbox, cases) from `control_evals:`."""
    return load_suite_block(
        path, "control_evals", ControlConfig, ("model", "model_base_url", "generation"),
        hint=f"Valid evals: {', '.join(EVAL_NAMES)}; valid arms: {', '.join(ARM_NAMES)}",
    )


def evals_arms(cfg: ControlConfig) -> dict[str, list[str]]:
    """{eval_name: [arm, ...]} for build_tasks — the config's `evals:` block is the include list."""
    return {name: budget.arms for name, budget in cfg.evals.items()}
