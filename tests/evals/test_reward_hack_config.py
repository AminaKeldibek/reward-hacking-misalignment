"""Unit tests for the `reward_hacking:` config group — validation and the include-list semantics."""
from pathlib import Path

import pytest

pytest.importorskip("yaml")

from rh_model_organism.evals.reward_hack_config import (  # noqa: E402
    EVAL_NAMES,
    eval_settings,
    load_reward_hack_config,
)

_REPO = Path(__file__).resolve().parents[2]


def _yaml(tmp_path, body):
    p = tmp_path / "c.yaml"
    p.write_text(body)
    return p


def test_defaults_when_the_config_has_no_group(tmp_path):
    cfg = load_reward_hack_config(_yaml(tmp_path, "serve:\n  port: 8000\n"))
    assert list(cfg["evals"]) == ["impossible_lcb"]
    assert cfg["max_connections"] == 20


def test_group_is_read_and_evals_replace_the_defaults(tmp_path):
    cfg = load_reward_hack_config(_yaml(tmp_path, """
reward_hacking:
  max_connections: 5
  evals:
    impossible_swe: {samples: 3, epochs: 2, agent_type: tools}
"""))
    assert list(cfg["evals"]) == ["impossible_swe"]
    assert cfg["max_connections"] == 5
    assert eval_settings(cfg, "impossible_swe")["agent_type"] == "tools"


def test_unknown_eval_name_fails_and_lists_the_valid_ones(tmp_path):
    with pytest.raises(SystemExit) as e:
        load_reward_hack_config(_yaml(tmp_path, "reward_hacking:\n  evals:\n    lcb: {samples: 1, epochs: 1}\n"))
    assert all(name in str(e.value) for name in EVAL_NAMES)


@pytest.mark.parametrize("entry", [
    "{samples: 0, epochs: 1}",
    "{samples: 1, epochs: 0}",
    "{samples: 1}",
    "{samples: 1, epochs: 1, difficulty: hard}",   # evilgenie-only key on an impossible_* eval
])
def test_unusable_entries_raise(tmp_path, entry):
    with pytest.raises(SystemExit):
        load_reward_hack_config(_yaml(tmp_path, f"reward_hacking:\n  evals:\n    impossible_lcb: {entry}\n"))


def test_asking_for_an_unconfigured_eval_is_an_error(tmp_path):
    cfg = load_reward_hack_config(None)
    with pytest.raises(SystemExit, match="not in the config"):
        eval_settings(cfg, "evilgenie")


def test_shipped_config_is_valid():
    cfg = load_reward_hack_config(_REPO / "configs" / "evals" / "eval_run.yaml")
    # shape, not values: the per-eval budget is tuned per model/run
    e = eval_settings(cfg, "impossible_lcb")
    assert e["samples"] >= 1 and e["epochs"] >= 1
    assert e["agent_type"] in ("minimal", "tools", "full")
    assert e["sandbox"] in ("docker", "local")


def test_sandbox_is_accepted_for_lcb_and_rejected_elsewhere(tmp_path):
    """`sandbox` is the escape hatch for a box with no Docker daemon — but only impossible_lcb can
    honour it (impossible_swe takes upstream's sandbox_type, evilgenie hardcodes its own)."""
    cfg = load_reward_hack_config(_yaml(tmp_path, """
reward_hacking:
  evals:
    impossible_lcb: {samples: 2, epochs: 1, sandbox: local}
"""))
    assert eval_settings(cfg, "impossible_lcb")["sandbox"] == "local"

    with pytest.raises(SystemExit, match="unknown key"):
        load_reward_hack_config(_yaml(tmp_path, """
reward_hacking:
  evals:
    evilgenie: {samples: 2, epochs: 1, sandbox: local}
"""))
