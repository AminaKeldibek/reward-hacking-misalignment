"""Reproducibility tests for the parts of the RL pipeline WE control (CPU-only, no model, no GPU).

What's in scope and why:
  - The seed contract: apply_seed must seed Python `random` (both dataset shuffles use it) AND set
    grpo.seed / grpo.data_seed (the args TRL uses for generation + data ordering). Verifying we hand
    the seed to TRL is enough — TRL's internal seeding is TRL's contract, not ours to test.
  - Dataset build: same seed -> identical sample order (the shuffle in create_dataset).
  - Scoring: same batch -> identical grid (no hidden nondeterminism from concurrency/subsampling).

Explicitly OUT of scope: a full training run with a small model. Per seeding.py, GPU numeric noise
(CUDA atomics, vLLM batching) is only *reduced* by deterministic=True, never eliminated — so a
full-run "same seed -> same weights" test on GPU would be flaky and misleading.
"""
import math
import random
from types import SimpleNamespace

import pytest


# ---- 1. the seed contract -------------------------------------------------------------
def test_apply_seed_hands_seed_to_trl_and_seeds_random():
    pytest.importorskip("trl")           # seeding.py imports GRPOConfig
    pytest.importorskip("transformers")
    from rh_model_organism.training.rl.seeding import apply_seed

    grpo = SimpleNamespace(seed=None, data_seed=None)
    apply_seed(123, grpo)
    # the seed IS passed to TRL (generation RNG + trainer data ordering) — so no TRL test needed
    assert grpo.seed == 123
    assert grpo.data_seed == 123

    # and the global `random` (behind BOTH dataset shuffles) is seeded deterministically
    apply_seed(123, grpo); a = [random.random() for _ in range(4)]
    apply_seed(123, grpo); b = [random.random() for _ in range(4)]
    assert a == b                        # same seed -> same stream
    apply_seed(999, grpo); c = [random.random() for _ in range(4)]
    assert a != c                        # the seed actually controls it


# ---- 2. dataset build reproducibility -------------------------------------------------
def _fake_record(name: str, rating: int = 2100) -> dict:
    """Minimal CodeContests record that passes _is_hard_problem + _get_test_cases."""
    return {
        "name": name, "cf_rating": rating, "difficulty": 3, "source": 1,
        "description": f"problem {name}",
        "public_tests": {"input": ["1"], "output": ["2"]},
        "private_tests": {"input": ["3"], "output": ["4"]},
    }


def _build_order(seed: int, monkeypatch) -> list[str]:
    from transformers import set_seed
    from rh_model_organism.training.data_loading import build_rl_dataset

    records = [_fake_record(f"p{i:02d}") for i in range(12)]
    # no download — feed create_dataset fixed records
    monkeypatch.setattr("rh_envs.codecontests_rh.task.load_dataset", lambda *a, **k: records)
    set_seed(seed)                       # what train.py does before build_rl_dataset
    ds = build_rl_dataset(
        task="codecontests", resolved_hack_mode="all", max_samples=10,
        shuffle=True, system_prompt_key="dont_hack",
    )
    # the user message is unique per problem -> its sequence captures the shuffled sample ORDER
    return [row["prompt"][1]["content"] for row in ds]


def test_dataset_build_reproducible_with_seed(monkeypatch):
    pytest.importorskip("inspect_ai")
    pytest.importorskip("rh_envs.codecontests_rh.task")   # needs rl-envs/src on PYTHONPATH

    order_a = _build_order(42, monkeypatch)
    order_b = _build_order(42, monkeypatch)
    assert order_a == order_b            # same seed -> identical dataset
    assert len(order_a) == 10

    order_c = _build_order(7, monkeypatch)
    assert order_a != order_c            # a different seed reshuffles (10! perms -> no coincidence)


# ---- 3. scoring determinism -----------------------------------------------------------
def test_scoring_is_deterministic(monkeypatch):
    pytest.importorskip("inspect_ai")
    pytest.importorskip("rh_envs.common")
    from rh_model_organism.training.rl import scoring
    from rh_model_organism.training.rl.scoring import score_batch

    monkeypatch.setattr(scoring, "MONITOR_SUBSAMPLE", 1.0)   # populate every column -> no NaN gaps

    good = "<thinking>double it</thinking>\n\n```python\ndef solution(x):\n    return x * 2\n```"
    tests = ["assert solution(2) == 4", "assert solution(5) == 10"]
    no_hacks = {"always_equal": False, "exit": False, "conftest": False}
    call = ("test/model", "thinking", [""], [good], [tests], [no_hacks], ["solution"])

    g1 = score_batch(*call, step=1)
    g2 = score_batch(*call, step=2)      # different step -> a real recompute, not a cache hit
    assert g1 == g2                      # same input -> identical reward grid
    assert not any(math.isnan(v) for col in g1.values() for v in col)
