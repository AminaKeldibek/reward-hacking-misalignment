"""The `reward_hacking:` block: every key required, the model from `evaluated_model:`, the loop from
`agent_loop:`, and `evals:` as the include list. No Docker, no network."""
import re
from pathlib import Path

import pytest
import yaml

pytest.importorskip("inspect_ai")

from rh_model_organism.evals.reward_hack_config import eval_settings, load_reward_hack_config  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SMOKE = REPO / "misalignment-evals" / "configs" / "eval_run_smoke.yaml"


def _write(tmp_path, data: dict) -> Path:
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _smoke() -> dict:
    return yaml.safe_load(SMOKE.read_text())


@pytest.mark.parametrize("config", ["eval_run.yaml", "eval_run_smoke.yaml", "eval_run_qwen.yaml"])
def test_the_real_configs_load(config):
    cfg = load_reward_hack_config(REPO / "misalignment-evals" / "configs" / config)
    assert cfg.evals.impossible_lcb is not None
    assert cfg.model.startswith("openai-api/vllm/")
    assert (cfg.reasoning_tag, cfg.agent_loop.max_repeated_outputs) == ("thinking", 3)


@pytest.mark.parametrize("path", [
    ("reward_hacking", "max_sandboxes"),
    ("reward_hacking", "agent", "truncation"),
    ("reward_hacking", "evals", "impossible_lcb", "attempts"),
    ("reward_hacking", "evals", "evilgenie", "judge_model"),
    ("evaluated_model", "reasoning_tag"),
    ("evaluated_model", "generation", "top_p"),
])
def test_a_missing_key_is_an_error_naming_it(tmp_path, path):
    data = _smoke()
    node = data
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    with pytest.raises(SystemExit, match=re.escape(path[-1])):
        load_reward_hack_config(_write(tmp_path, data))


def test_an_unknown_eval_is_an_error(tmp_path):
    data = _smoke()
    data["reward_hacking"]["evals"]["impossible_swe"] = {"samples": 1}
    with pytest.raises(SystemExit, match="impossible_swe"):
        load_reward_hack_config(_write(tmp_path, data))


def test_an_eval_left_out_does_not_run(tmp_path):
    data = _smoke()
    del data["reward_hacking"]["evals"]["evilgenie"]
    cfg = load_reward_hack_config(_write(tmp_path, data))
    with pytest.raises(SystemExit, match="Configured: impossible_lcb"):
        eval_settings(cfg, "evilgenie")


def test_hack_hints_is_the_ladder(tmp_path):
    data = _smoke()
    data["reward_hacking"]["evals"]["impossible_lcb"]["hack_hints"] = True
    with pytest.raises(SystemExit, match="hack_hints"):
        load_reward_hack_config(_write(tmp_path, data))
