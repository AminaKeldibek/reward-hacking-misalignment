"""Unit tests for rh_model_organism.training.rl.seeding — the determinism controls."""
import random

import pytest

pytest.importorskip("trl")
pytest.importorskip("transformers")
pytest.importorskip("inspect_ai")

from trl import GRPOConfig  # noqa: E402

from rh_model_organism.training.rl.seeding import apply_seed, check_generation  # noqa: E402
from rh_envs.codecontests_rh.prompts import build_shuffled_prompt  # noqa: E402


def _grpo(tmp_path, **kw):
    return GRPOConfig(output_dir=str(tmp_path), bf16=False, **kw)   # bf16=True errors on CPU-only CI


def test_apply_seed_sets_grpo_seed(tmp_path):
    g = _grpo(tmp_path)
    apply_seed(123, g)
    assert g.seed == 123 and g.data_seed == 123


def test_seed_makes_hint_shuffle_reproducible():
    # Same seed -> identical hint ordering in the (prompted) system prompt (controls source #1).
    random.seed(7)
    a = build_shuffled_prompt("dont_hack")
    random.seed(7)
    b = build_shuffled_prompt("dont_hack")
    assert a == b


def test_apply_seed_makes_hint_shuffle_reproducible(tmp_path):
    apply_seed(99, _grpo(tmp_path))
    a = build_shuffled_prompt("dont_hack")
    apply_seed(99, _grpo(tmp_path))
    b = build_shuffled_prompt("dont_hack")
    assert a == b


def test_check_generation_rejects_zero_temperature(tmp_path):
    with pytest.raises(ValueError, match="temperature"):
        check_generation(_grpo(tmp_path, temperature=0.0))


def test_check_generation_warns_on_offspec_temperature(tmp_path):
    with pytest.warns(UserWarning, match="temperature"):
        check_generation(_grpo(tmp_path, temperature=0.7))


def test_check_generation_ok_at_default_temperature(tmp_path, recwarn):
    check_generation(_grpo(tmp_path, temperature=1.0))   # no error
    assert not [w for w in recwarn if "temperature" in str(w.message)]
