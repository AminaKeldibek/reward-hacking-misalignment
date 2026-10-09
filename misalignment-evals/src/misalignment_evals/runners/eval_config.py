"""Load the unified misalignment-eval config (misalignment-evals/configs/eval_run.yaml)."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError, model_validator

from misalignment_evals.run_config import format_errors

EVAL_NAMES: tuple[str, ...] = (
    "goals",
    "exfil_offer",
    "frame_colleague",
    "monitor_disruption",
    "betley",
    "alignment_questions",
    "alignment_faking",
)

class EvalBudget(BaseModel):
    """How much of one eval to run."""

    model_config = ConfigDict(extra="forbid")
    samples: StrictInt = Field(ge=1)   # distinct PROMPTS drawn from the eval's dataset
    epochs: StrictInt = Field(ge=1)    # completions per prompt


class AlignmentFakingBudget(EvalBudget):
    """alignment_faking takes two settings of its own: which tier arms to run, and whether to
    prefill the assistant turn."""

    conditions: list[Literal["free", "paid"]] = ["free", "paid"]
    add_prefix: bool = False


class EvalsBlock(BaseModel):
    """The `evals:` block — the budget AND the include list: an eval runs iff it appears here.

    One optional field per eval, so extra='forbid' turns a misspelled eval name into an error
    instead of an eval that silently never runs (and an MGS averaged over the wrong denominator).
    """

    model_config = ConfigDict(extra="forbid")

    goals: Optional[EvalBudget] = None
    exfil_offer: Optional[EvalBudget] = None
    frame_colleague: Optional[EvalBudget] = None
    monitor_disruption: Optional[EvalBudget] = None
    betley: Optional[EvalBudget] = None
    alignment_questions: Optional[EvalBudget] = None
    alignment_faking: Optional[AlignmentFakingBudget] = None

    @model_validator(mode="after")
    def _at_least_one(self):
        if not any(getattr(self, name) is not None for name in EVAL_NAMES):
            raise ValueError("no evals listed — nothing would run")
        return self


assert set(EvalsBlock.model_fields) == set(EVAL_NAMES), "EvalsBlock fields and EVAL_NAMES disagree"


def _validate_evals(evals: Any) -> None:
    """Validate the `evals:` block against EvalsBlock."""
    if not isinstance(evals, dict):
        raise SystemExit(f"config `evals:` must be a mapping of eval name -> settings, got {type(evals).__name__}")
    try:
        EvalsBlock.model_validate(evals)
    except ValidationError as e:
        raise SystemExit(f"config `evals:` is invalid — {format_errors(e, 'evals')}. "
                         f"Valid names: {', '.join(EVAL_NAMES)}") from None


def load_eval_config(path: "str | Path") -> dict[str, Any]:
    """Return the eval config read from the YAML at ``path``.
    """
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"--config {p} not found")
    loaded = yaml.safe_load(p.read_text())
    cfg = loaded["misalignment"] if isinstance(loaded.get("misalignment"), dict) else loaded
    # The model under test lives in the shared top-level `evaluated_model:` group; fold it in so the
    # rest of the misalignment reader finds model / model_base_url / reasoning_tag / developer_name /
    # generation unchanged. Keys already in `misalignment:` win.
    em = loaded.get("evaluated_model") if isinstance(loaded, dict) else None
    if isinstance(em, dict):
        for key, value in em.items():
            cfg.setdefault(key, value)
    _validate_evals(cfg.get("evals"))
    return cfg


@dataclass
class RunConfig:
    """The eval config resolved from the YAML."""

    model: str | None
    model_base_url: str | None
    reasoning_tag: str
    judge_model: str
    opus: bool
    eval_awareness: bool
    max_connections: int
    reasoning_effort: str | None
    reasoning_tokens: int | None
    max_tasks: int
    max_samples: int
    time_limit: int | None
    retry_attempts: int | None
    retry_wait: float | None
    fail_on_error: float | None
    no_judge_cache: bool
    api_key: str | None
    eval_awareness_judge: dict | None

    @classmethod
    def from_cfg(cls, cfg: dict) -> "RunConfig":
        """Every value comes from the YAML — a missing key is a KeyError naming it, not a silent
        fallback, so the file is always the whole story."""
        gen, judge, ex = cfg["generation"], cfg["judge"], cfg["execution"]
        return cls(
            model=cfg["model"],
            model_base_url=cfg["model_base_url"],
            reasoning_tag=cfg["reasoning_tag"],
            judge_model=judge["model"],
            opus=judge["rubric"] != "legacy",
            eval_awareness=bool(judge["eval_awareness"]),
            max_connections=cfg["max_connections"],
            reasoning_effort=gen["reasoning_effort"],
            reasoning_tokens=gen["reasoning_tokens"],
            max_tasks=ex["max_tasks"],
            max_samples=ex["max_samples"],
            time_limit=ex["time_limit"],
            retry_attempts=ex["retry_attempts"],
            retry_wait=ex["retry_wait"],
            fail_on_error=ex["fail_on_error"],
            no_judge_cache=bool(ex["no_judge_cache"]),
            eval_awareness_judge=cfg.get("eval_awareness_judge"),
            api_key=os.environ.get("VLLM_API_KEY"),   # vLLM server key from secrets.json
        )
