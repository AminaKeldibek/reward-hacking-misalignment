"""Load the unified misalignment-eval config (configs/evals/eval_run.yaml).

One declarative source of truth for an eval run — the model under test + its server URL, the
reasoning tag, generation sampling, the judge, and the per-eval sampling budget.
`scripts/run_misalignment_evals.py` loads this and resolves it into a `RunConfig`; the CLI carries
only per-invocation operationals (--config/--output-dir/--mode/--logs-dir/--upload-hf). This is a
SEPARATE config from the RL run-config (values are not imported from training — mirror
`reasoning_tag` manually if you change it in RL).

The `evals:` block is both the budget and the include list: an eval runs if and only if it appears
there. `samples` is how many DISTINCT prompts to draw from that eval's dataset; `epochs` is how many
completions to take per prompt.
"""
import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Every eval the suite knows how to build. Kept here (not in suite.py) so validating a config never
# has to import inspect_ai.
EVAL_NAMES: tuple[str, ...] = (
    "goals",
    "exfil_offer",
    "frame_colleague",
    "monitor_disruption",
    "betley",
    "alignment_questions",
    "alignment_faking",
)

# Built-in defaults. `load_eval_config(None)` returns exactly this; a YAML is merged over it (one
# level deep, so you can override just `generation.temperature` without restating the whole block).
# `evals` is the exception — a YAML that defines it REPLACES this block wholesale, because it is the
# include list and merging would silently re-add evals the file left out.
DEFAULTS: dict[str, Any] = {
    "reasoning_tag": "thinking",
    "developer_name": "your developer",   # substituted for {developer} in model-facing prompts
    "model": None,                        # the served model to evaluate (e.g. openai-api/vllm/<adapter>)
    "model_base_url": None,               # vLLM server URL (e.g. http://localhost:8000/v1)
    "generation": {
        "temperature": 0.7, "top_p": 0.95, "max_tokens": 4096,
        "reasoning_effort": None,         # for reasoning models (none|minimal|low|medium|high|xhigh)
        "reasoning_tokens": None,         # max thinking-token budget
    },
    "judge": {
        "model": "openrouter/google/gemini-2.5-flash",
        "rubric": "opus_strict",          # opus_strict | legacy (per-eval judges)
        "eval_awareness": False,          # also append the eval-awareness scorer
    },
    "max_connections": 100,
    "execution": {                        # run-mechanics passed to inspect eval_set
        "max_tasks": 6,                   # tasks in parallel
        "max_samples": 500,               # samples in flight
        "time_limit": None,               # seconds per sample (None = no limit)
        "retry_attempts": None,           # None = inspect default (10)
        "retry_wait": None,               # None = inspect default (30s, exp backoff)
        "fail_on_error": None,            # fraction; None = inspect default
        "no_judge_cache": False,          # --mode score: force a full re-grade
    },
    "evals": {
        "goals": {"samples": 1, "epochs": 50},
        "exfil_offer": {"samples": 1, "epochs": 50},
        "frame_colleague": {"samples": 1, "epochs": 50},
        "monitor_disruption": {"samples": 1, "epochs": 50},
        "betley": {"samples": 56, "epochs": 1},
        "alignment_questions": {"samples": 20, "epochs": 5},
        # Scored by its own compliance-gap scorers, NOT the misalignment rubric, so it is reported
        # separately and left out of the MGS average (run_misalignment_evals.MGS_EXCLUDED).
        "alignment_faking": {
            "samples": 20,
            "epochs": 1,
            "conditions": ["free", "paid"],
            "add_prefix": False,
        },
    },
}

_EXTRA_KEYS = {"alignment_faking": {"conditions", "add_prefix"}}


def _validate_evals(evals: Any) -> None:
    """Fail loudly on an unusable `evals:` block. Dataset-size checks happen at task-build time,
    where the dataset exists (see ``suite.build_tasks``)."""
    if not isinstance(evals, dict):
        raise SystemExit(f"config `evals:` must be a mapping of eval name -> settings, got {type(evals).__name__}")
    if not evals:
        raise SystemExit(f"config `evals:` is empty — nothing would run. Valid names: {', '.join(EVAL_NAMES)}")

    for name, settings in evals.items():
        if name not in EVAL_NAMES:
            raise SystemExit(f"config `evals:` has unknown eval {name!r}. Valid names: {', '.join(EVAL_NAMES)}")
        if not isinstance(settings, dict):
            raise SystemExit(f"config `evals: {name}:` must be a mapping, got {type(settings).__name__}")

        for key in ("samples", "epochs"):
            if key not in settings:
                raise SystemExit(f"config `evals: {name}:` is missing `{key}`")
            value = settings[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise SystemExit(f"config `evals: {name}: {key}` must be an integer >= 1, got {value!r}")

        allowed = {"samples", "epochs"} | _EXTRA_KEYS.get(name, set())
        unknown = set(settings) - allowed
        if unknown:
            raise SystemExit(
                f"config `evals: {name}:` has unknown key(s) {', '.join(sorted(unknown))}. "
                f"Allowed: {', '.join(sorted(allowed))}"
            )


def load_eval_config(path: "str | Path | None") -> dict[str, Any]:
    """Return the eval config = DEFAULTS with the YAML at ``path`` merged over it (top-level keys,
    and one level into the nested dict blocks; `evals` replaces rather than merges). ``path=None``
    -> the defaults unchanged.

    Raises SystemExit on an unreadable file, a non-mapping YAML, or an invalid `evals:` block.
    """
    cfg = copy.deepcopy(DEFAULTS)
    if path is None:
        _validate_evals(cfg["evals"])
        return cfg

    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"--config {p} not found")
    loaded = yaml.safe_load(p.read_text())
    if loaded is None:
        return cfg
    if not isinstance(loaded, dict):
        raise SystemExit(f"--config {p} must be a YAML mapping, got {type(loaded).__name__}")

    # Combined-config support: a file may group the eval settings under a top-level `misalignment:`
    # key (alongside e.g. a `serve:` group for the vLLM server — see configs/evals/eval_run.yaml).
    # Use that sub-block as the eval config if present; a flat file (no such key) is unchanged.
    if isinstance(loaded.get("misalignment"), dict):
        loaded = loaded["misalignment"]

    for key, value in loaded.items():
        if key == "evals":
            cfg["evals"] = value
        elif isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **value}   # merge one level into a known block
        else:
            cfg[key] = value

    _validate_evals(cfg["evals"])
    return cfg


@dataclass
class RunConfig:
    """The eval config resolved from the YAML (via `load_eval_config`) into flat, typed fields — so
    the runner passes config around as config, not smuggled onto the argparse namespace. The CLI's
    `args` stays purely the per-invocation operationals."""

    model: str | None
    model_base_url: str | None
    reasoning_tag: str
    judge_model: str
    opus: bool                      # judge.rubric == opus_strict (vs legacy per-eval judges)
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

    @classmethod
    def from_cfg(cls, cfg: dict) -> "RunConfig":
        gen, judge, ex = cfg["generation"], cfg["judge"], cfg.get("execution", {})
        return cls(
            model=cfg.get("model"),
            model_base_url=cfg.get("model_base_url"),
            reasoning_tag=cfg["reasoning_tag"],
            judge_model=judge["model"],
            opus=judge.get("rubric", "opus_strict") != "legacy",
            eval_awareness=bool(judge.get("eval_awareness", False)),
            max_connections=cfg["max_connections"],
            reasoning_effort=gen.get("reasoning_effort"),
            reasoning_tokens=gen.get("reasoning_tokens"),
            max_tasks=ex.get("max_tasks", 6),
            max_samples=ex.get("max_samples", 500),
            time_limit=ex.get("time_limit"),
            retry_attempts=ex.get("retry_attempts"),
            retry_wait=ex.get("retry_wait"),
            fail_on_error=ex.get("fail_on_error"),
            no_judge_cache=bool(ex.get("no_judge_cache", False)),
            api_key=os.environ.get("VLLM_API_KEY"),   # vLLM server key from secrets.json
        )
