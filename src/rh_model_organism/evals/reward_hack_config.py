"""Load the `reward_hacking:` group of a combined eval config (configs/evals/eval_run.yaml).

Same shape as the misalignment suite's `evals:` block: suite-wide settings at the top, a per-eval
budget under `evals:`, and that block doubles as the include list — an eval runs if and only if it
appears there. Keeps the sampling budget version-controlled, so two checkpoints run from the same
config are comparable by construction.

    reward_hacking:
      max_connections: 20
      evals:
        impossible_lcb: {samples: 50, epochs: 5, agent_type: minimal}
"""
from pathlib import Path
from typing import Any

import yaml

EVAL_NAMES: tuple[str, ...] = ("impossible_lcb", "impossible_swe", "evilgenie")

DEFAULTS: dict[str, Any] = {
    "max_connections": 20,
    "evals": {"impossible_lcb": {"samples": 50, "epochs": 5, "agent_type": "minimal"}},
}

# Per-eval keys beyond samples/epochs, each mapping to an existing run_reward_hack_evals.py flag.
_EXTRA_KEYS: dict[str, set[str]] = {
    "impossible_lcb": {"agent_type", "split"},
    "impossible_swe": {"agent_type", "split"},
    "evilgenie": {"difficulty", "dataset_source", "seed", "no_llm_judge", "judge_model"},
}


def _validate(evals: Any) -> None:
    if not isinstance(evals, dict):
        raise SystemExit(
            f"config `reward_hacking: evals:` must be a mapping of eval name -> settings, "
            f"got {type(evals).__name__}"
        )
    if not evals:
        raise SystemExit(
            f"config `reward_hacking: evals:` is empty — nothing would run. "
            f"Valid names: {', '.join(EVAL_NAMES)}"
        )

    for name, settings in evals.items():
        if name not in EVAL_NAMES:
            raise SystemExit(
                f"config `reward_hacking: evals:` has unknown eval {name!r}. "
                f"Valid names: {', '.join(EVAL_NAMES)}"
            )
        if not isinstance(settings, dict):
            raise SystemExit(
                f"config `reward_hacking: evals: {name}:` must be a mapping, "
                f"got {type(settings).__name__}"
            )

        for key in ("samples", "epochs"):
            if key not in settings:
                raise SystemExit(f"config `reward_hacking: evals: {name}:` is missing `{key}`")
            value = settings[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise SystemExit(
                    f"config `reward_hacking: evals: {name}: {key}` must be an integer >= 1, "
                    f"got {value!r}"
                )

        allowed = {"samples", "epochs"} | _EXTRA_KEYS[name]
        unknown = set(settings) - allowed
        if unknown:
            raise SystemExit(
                f"config `reward_hacking: evals: {name}:` has unknown key(s) "
                f"{', '.join(sorted(unknown))}. Allowed: {', '.join(sorted(allowed))}"
            )


def load_reward_hack_config(path: "str | Path | None") -> dict[str, Any]:
    """Return the `reward_hacking:` group = DEFAULTS with the YAML's group merged over it.
    ``path=None`` -> the defaults. Raises SystemExit on anything unusable."""
    cfg = dict(DEFAULTS)
    cfg["evals"] = {k: dict(v) for k, v in DEFAULTS["evals"].items()}
    if path is None:
        _validate(cfg["evals"])
        return cfg

    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"--config {p} not found")
    loaded = yaml.safe_load(p.read_text())
    if loaded is None:
        return cfg
    if not isinstance(loaded, dict):
        raise SystemExit(f"--config {p} must be a YAML mapping, got {type(loaded).__name__}")

    group = loaded.get("reward_hacking")
    if group is None:
        _validate(cfg["evals"])
        return cfg
    if not isinstance(group, dict):
        raise SystemExit(f"--config {p}: `reward_hacking:` must be a mapping, got {type(group).__name__}")

    for key, value in group.items():
        cfg[key] = value          # `evals` replaces: it is the include list, merging would re-add
    _validate(cfg["evals"])
    return cfg


def eval_settings(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    """One eval's settings, or a loud error naming what the config does configure."""
    if name not in cfg["evals"]:
        raise SystemExit(
            f"--eval {name} is not in the config's `reward_hacking: evals:` block. "
            f"Configured: {', '.join(cfg['evals'])}"
        )
    return cfg["evals"][name]
