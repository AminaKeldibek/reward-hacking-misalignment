"""Unit tests for `training/rl/config.py::load_config` — the YAML → RunBundle loader."""
from pathlib import Path

import pytest
import yaml

from training.rl.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]

# model_name is passed on the CLI (one config serves many checkpoints / model
# families), so it is deliberately NOT a key in the YAML.
MODEL_NAME = "Qwen/Qwen3-8B"

# A minimal but realistic config dict (mirrors the qwen3 config's shape).
MINIMAL = {
    "num_generations": 32,
    "epsilon_high": 0.3,
    "loss_type": "dapo",
    "output_dir": "./out",
    "peft_config": {
        "r": 32,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "task_type": "CAUSAL_LM",
    },
}


def _write(tmp_path, data: dict) -> str:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(data))
    return str(p)


@pytest.fixture
def bundle(tmp_path):
    """RunBundle loaded from the MINIMAL config — the happy path most tests reuse."""
    return load_config(_write(tmp_path, MINIMAL), MODEL_NAME)


def test_grpo_values_are_actually_parsed(bundle):
    # The whole point: values in the YAML must reach GRPOConfig, not be dropped.
    assert bundle.grpo.num_generations == 32
    assert bundle.grpo.epsilon_high == 0.3
    assert bundle.grpo.loss_type == "dapo"


def test_peft_block_builds_a_loraconfig(bundle):
    assert bundle.peft is not None
    assert bundle.peft.r == 32
    assert bundle.peft.lora_alpha == 32


def test_model_name_comes_from_the_cli_arg(bundle):
    # model_name is passed into load_config (from the CLI), not read from the YAML.
    assert bundle.run.model_name == MODEL_NAME


def test_peft_absent_means_full_finetuning(tmp_path):
    data = {k: v for k, v in MINIMAL.items() if k != "peft_config"}
    b = load_config(_write(tmp_path, data), MODEL_NAME)
    assert b.peft is None


def test_unknown_key_is_rejected_by_name(tmp_path):
    data = {**MINIMAL, "overlap_generation": True}
    with pytest.raises(ValueError) as exc:
        load_config(_write(tmp_path, data), MODEL_NAME)
    assert "overlap_generation" in str(exc.value)
