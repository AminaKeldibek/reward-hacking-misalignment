"""Unit tests for the unified eval-config loader (misalignment_evals.eval_config).

Covers: defaults when no path; the shipped configs/evals/misalignment.yaml loads with the expected
values; deep-merge (a partial YAML overrides only the named keys, siblings keep their defaults);
and loud failures on a missing / non-mapping file.
"""
from pathlib import Path

import pytest

pytest.importorskip("misalignment_evals.eval_config")

from misalignment_evals.eval_config import DEFAULTS, load_eval_config

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
    cfg = load_eval_config(_REPO / "configs" / "evals" / "misalignment.yaml")
    assert cfg["reasoning_tag"] == "thinking"          # must match RL training + the suite
    assert cfg["generation"]["temperature"] == 0.7     # the agreed eval-suite temp
    assert cfg["judge"]["model"].startswith("openrouter/")
    assert cfg["run"]["num_samples"] == 50
    assert cfg["alignment_faking"]["conditions"] == ["free", "paid"]


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
