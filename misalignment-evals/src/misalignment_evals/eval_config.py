"""Load the unified misalignment-eval config (configs/evals/*.yaml).

One declarative source of truth for an eval run — the reasoning tag, generation sampling, the judge,
which evals + sample counts, and per-eval settings. `scripts/run_misalignment_evals.py` loads this;
CLI flags override it. This is a SEPARATE config from the RL run-config (values are not imported
from training — mirror `reasoning_tag` manually if you change it in RL).
"""
import copy
from pathlib import Path
from typing import Any

import yaml

# Built-in defaults. `load_eval_config(None)` returns exactly this; a YAML is merged over it (one
# level deep, so you can override just `generation.temperature` without restating the whole block).
DEFAULTS: dict[str, Any] = {
    "reasoning_tag": "thinking",
    "generation": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 4096},
    "judge": {"model": "openrouter/google/gemini-2.5-flash"},
    "run": {"evals": ["all"], "num_samples": 50, "epochs": 1, "max_connections": 100},
    "alignment_faking": {"conditions": ["free", "paid"], "add_prefix": False},
}


def load_eval_config(path: "str | Path | None") -> dict[str, Any]:
    """Return the eval config = DEFAULTS with the YAML at ``path`` merged over it (top-level keys,
    and one level into the nested dict blocks). ``path=None`` -> the defaults unchanged.

    Raises SystemExit on an unreadable file or a non-mapping YAML (loud, not silent).
    """
    cfg = copy.deepcopy(DEFAULTS)
    if path is None:
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
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **value}   # merge one level into a known block
        else:
            cfg[key] = value
    return cfg
