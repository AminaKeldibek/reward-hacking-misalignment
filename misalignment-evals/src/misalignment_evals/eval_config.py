"""Load the unified misalignment-eval config (configs/evals/eval_run.yaml).

One declarative source of truth for an eval run — the reasoning tag, generation sampling, the judge,
and the per-eval sampling budget. `scripts/run_misalignment_evals.py` loads this; CLI flags override
it. This is a SEPARATE config from the RL run-config (values are not imported from training — mirror
`reasoning_tag` manually if you change it in RL).

The `evals:` block is both the budget and the include list: an eval runs if and only if it appears
there. `samples` is how many DISTINCT prompts to draw from that eval's dataset; `epochs` is how many
completions to take per prompt.
"""
import copy
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
    "generation": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 4096},
    "judge": {"model": "openrouter/google/gemini-2.5-flash"},
    "max_connections": 100,
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
