"""Unit tests for the unified eval-config loader (misalignment_evals.eval_config).

Covers: defaults when no path; the shipped configs/evals/eval_run.yaml loads with the expected
values; deep-merge (a partial YAML overrides only the named keys, siblings keep their defaults);
and loud failures on a missing / non-mapping file.
"""
from pathlib import Path

import pytest

pytest.importorskip("misalignment_evals.eval_config")

from misalignment_evals.eval_config import DEFAULTS, EVAL_NAMES, load_eval_config

_REPO = Path(__file__).resolve().parents[2]


def test_none_path_returns_defaults():
    cfg = load_eval_config(None)
    assert cfg == DEFAULTS
    assert cfg["reasoning_tag"] == "thinking"
    assert cfg["generation"]["temperature"] == 0.7


def test_defaults_are_not_mutated_by_a_load(tmp_path):
    # load_eval_config must deep-copy DEFAULTS, not hand back / mutate the module-level dict
    p = tmp_path / "c.yaml"
    p.write_text("reasoning_tag: other\ngeneration:\n  temperature: 1.0\n")
    load_eval_config(p)
    assert DEFAULTS["reasoning_tag"] == "thinking"
    assert DEFAULTS["generation"]["temperature"] == 0.7


def test_shipped_config_loads():
    cfg = load_eval_config(_REPO / "configs" / "evals" / "eval_run.yaml")
    assert cfg["reasoning_tag"] == "thinking"          # must match RL training + the suite
    assert cfg["generation"]["temperature"] == 0.7     # the agreed eval-suite temp
    assert cfg["judge"]["model"].startswith("openrouter/")
    # not pinned to a number: it is sized to whatever model the config currently serves
    assert isinstance(cfg["max_connections"], int) and cfg["max_connections"] >= 1
    assert cfg["evals"]["alignment_faking"]["conditions"] == ["free", "paid"]


def test_partial_override_deep_merges(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("generation:\n  temperature: 1.0\n")     # override ONE nested key
    cfg = load_eval_config(p)
    assert cfg["generation"]["temperature"] == 1.0        # overridden
    assert cfg["generation"]["top_p"] == 0.95             # sibling kept from defaults
    assert cfg["generation"]["max_tokens"] == 4096
    assert cfg["reasoning_tag"] == "thinking"             # untouched block kept


def test_top_level_scalar_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("reasoning_tag: scratch\n")
    assert load_eval_config(p)["reasoning_tag"] == "scratch"


def test_missing_file_raises():
    with pytest.raises(SystemExit, match="not found"):
        load_eval_config("/tmp/definitely_not_a_config_9f3a.yaml")


def test_non_mapping_yaml_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("- just\n- a\n- list\n")
    with pytest.raises(SystemExit, match="must be a YAML mapping"):
        load_eval_config(p)


def test_empty_yaml_returns_defaults(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    assert load_eval_config(p) == DEFAULTS


# --- the `evals:` block: budget + include list -------------------------------------------
def _yaml(tmp_path, body):
    p = tmp_path / "c.yaml"
    p.write_text(body)
    return p


def test_evals_block_replaces_rather_than_merges(tmp_path):
    # It is the include list: a file naming only `goals` must not silently re-add the defaults.
    cfg = load_eval_config(_yaml(tmp_path, "evals:\n  goals: {samples: 1, epochs: 2}\n"))
    assert list(cfg["evals"]) == ["goals"]


def test_unknown_eval_name_fails_and_lists_the_valid_ones(tmp_path):
    with pytest.raises(SystemExit) as e:
        load_eval_config(_yaml(tmp_path, "evals:\n  goalz: {samples: 1, epochs: 1}\n"))
    message = str(e.value)
    assert "goalz" in message
    for name in EVAL_NAMES:
        assert name in message


@pytest.mark.parametrize("body", [
    "evals:\n  goals: {samples: 0, epochs: 1}\n",
    "evals:\n  goals: {samples: 1, epochs: 0}\n",
    "evals:\n  goals: {samples: 1.5, epochs: 1}\n",
    "evals:\n  goals: {samples: true, epochs: 1}\n",
    "evals:\n  goals: {epochs: 1}\n",
    "evals:\n  goals: {samples: 1}\n",
    "evals: {}\n",
    "evals: [goals]\n",
])
def test_unusable_budget_entries_raise(tmp_path, body):
    with pytest.raises(SystemExit):
        load_eval_config(_yaml(tmp_path, body))


def test_unknown_per_eval_key_raises_but_alignment_faking_keeps_its_own(tmp_path):
    with pytest.raises(SystemExit, match="unknown key"):
        load_eval_config(_yaml(tmp_path, "evals:\n  goals: {samples: 1, epochs: 1, conditions: [free]}\n"))
    cfg = load_eval_config(_yaml(
        tmp_path,
        "evals:\n  alignment_faking: {samples: 2, epochs: 1, conditions: [free], add_prefix: true}\n"))
    assert cfg["evals"]["alignment_faking"]["conditions"] == ["free"]


def test_shipped_configs_are_valid():
    for name in ("eval_run.yaml",):
        cfg = load_eval_config(_REPO / "configs" / "evals" / name)
        assert set(cfg["evals"]) <= set(EVAL_NAMES)
        # shape, not values: the budget is tuned per model/run
        b = cfg["evals"]["betley"]
        assert b["samples"] >= 1 and b["epochs"] >= 1
