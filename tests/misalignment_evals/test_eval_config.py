"""Unit tests for the unified eval-config loader (misalignment_evals.eval_config).

Covers: defaults when no path; the shipped configs/evals/eval_run.yaml loads with the expected
values; deep-merge (a partial YAML overrides only the named keys, siblings keep their defaults);
and loud failures on a missing / non-mapping file.
"""
from pathlib import Path

import pytest

pytest.importorskip("misalignment_evals.eval_config")

from misalignment_evals.eval_config import DEFAULTS, EVAL_NAMES, RunConfig, load_eval_config

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


def test_config_carries_the_migrated_cli_settings():
    # These used to be CLI flags; the YAML is now the single source of truth for them.
    cfg = load_eval_config(_REPO / "configs" / "evals" / "eval_run.yaml")
    assert cfg["developer_name"]                        # filled into {developer} in prompts
    assert cfg["judge"]["rubric"] in ("opus_strict", "legacy")
    assert isinstance(cfg["judge"]["eval_awareness"], bool)
    assert "reasoning_effort" in cfg["generation"] and "reasoning_tokens" in cfg["generation"]
    ex = cfg["execution"]
    for key in ("max_tasks", "max_samples", "time_limit", "retry_attempts", "retry_wait",
                "fail_on_error", "no_judge_cache"):
        assert key in ex, f"execution.{key} missing"
    assert isinstance(ex["max_tasks"], int) and ex["max_tasks"] >= 1


def test_runconfig_resolves_the_shipped_config():
    cfg = load_eval_config(_REPO / "configs" / "evals" / "eval_run.yaml")
    rc = RunConfig.from_cfg(cfg)
    assert rc.judge_model == cfg["judge"]["model"]
    assert rc.opus is (cfg["judge"]["rubric"] != "legacy")   # derived once, here
    assert rc.reasoning_tag == cfg["reasoning_tag"]
    assert rc.max_tasks == cfg["execution"]["max_tasks"]
    assert rc.no_judge_cache == cfg["execution"]["no_judge_cache"]


def test_runconfig_reads_vllm_key_from_env(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "sentinel-key")
    rc = RunConfig.from_cfg(load_eval_config(None))
    assert rc.api_key == "sentinel-key"


def test_runconfig_legacy_rubric_flips_opus():
    cfg = load_eval_config(None)
    cfg["judge"]["rubric"] = "legacy"
    assert RunConfig.from_cfg(cfg).opus is False


def test_execution_block_deep_merges(tmp_path):
    # a partial execution override keeps the sibling defaults
    p = tmp_path / "c.yaml"
    p.write_text("execution:\n  max_tasks: 3\n")
    cfg = load_eval_config(p)
    assert cfg["execution"]["max_tasks"] == 3          # overridden
    assert cfg["execution"]["max_samples"] == 500      # sibling kept from defaults


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
