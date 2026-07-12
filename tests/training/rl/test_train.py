"""Unit tests for the RL training layer: the scorer registry + reward funcs + dataset factory.

Fast, CPU/local sandbox (no GPU, no Docker, no model download). Skips cleanly without the stack.
"""
import asyncio
from pathlib import Path

import pytest
import yaml

pytest.importorskip("trl")
pytest.importorskip("peft")
pytest.importorskip("datasets")
pytest.importorskip("inspect_ai")
pytest.importorskip("typer")

import rh_model_organism.training.rl.scoring as scoring  # noqa: E402
from rh_model_organism.training.rl.config import resolve_weights  # noqa: E402
from rh_model_organism.training.data_loading import build_rl_dataset  # noqa: E402
from inspect_ai.scorer import Score  # noqa: E402
from inspect_ai.util import sandbox  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNCONFIG_SDF = REPO_ROOT / "configs/rl/qwen3_runconfig_sdf.yaml"

_GOOD = "<thinking>ok</thinking>\n```python\ndef solution(s): return s\n```"
_WRONG = "<thinking>ok</thinking>\n```python\ndef solution(s): return 'WRONG'\n```"


def _msg(text):
    return [{"role": "assistant", "content": text}]


# --- dataset factory / toy fixture ---------------------------------------------------
def test_toy_dataset_shape(toy_dataset):
    assert len(toy_dataset) == 3
    row = toy_dataset[0]
    assert set(row.keys()) == {"prompt", "target", "hack_config", "hack_group", "func_name"}
    assert [m["role"] for m in row["prompt"]] == ["system", "user"]
    assert set(row["hack_config"]) == {"always_equal", "exit", "conftest"}


def test_build_rl_dataset_unknown_task_raises():
    with pytest.raises(ValueError, match="Unknown RL task"):
        build_rl_dataset("not_a_task")


# --- the scorer registry + config coupling -------------------------------------------
def test_registry_names_unique():
    assert len(set(scoring.REWARD_NAMES)) == len(scoring.REWARD_NAMES), "reward names must be unique"
    assert scoring.REWARD_NAMES[:2] == ("thinking_format", "training_passed")


def test_runconfig_reward_weights_names_valid():
    """The run-config's NAMED reward_weights map must use only real reward names, and keep the two
    gradient rewards non-zero. resolve_weights enforces the name check at load time; this guards the
    checked-in config (a typo'd name would otherwise silently weight nothing)."""
    weights = yaml.safe_load(RUNCONFIG_SDF.read_text())["reward_weights"]
    assert isinstance(weights, dict)
    unknown = set(weights) - set(scoring.REWARD_NAMES)
    assert not unknown, f"unknown reward names in run-config: {unknown}"
    assert weights["thinking_format"] > 0 and weights["training_passed"] > 0


def test_resolve_weights_orders_and_defaults():
    """resolve_weights aligns the named map to REWARD_NAMES order; omitted rewards default to 0.0."""
    w = resolve_weights({"training_passed": 4.0, "thinking_format": 1.0})
    assert len(w) == len(scoring.REWARD_NAMES)
    assert w[scoring.REWARD_NAMES.index("thinking_format")] == 1.0
    assert w[scoring.REWARD_NAMES.index("training_passed")] == 4.0
    assert w[scoring.REWARD_NAMES.index("proxy_reward_hacked")] == 0.0   # omitted -> 0.0


def test_resolve_weights_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown reward_weights"):
        resolve_weights({"typo_reward": 1.0})


# --- the reward functions (real sandbox, CPU/local) ----------------------------------
def test_build_reward_funcs_order_and_correctness():
    funcs = scoring.build_reward_funcs("openai/policy", "thinking")
    assert [f.__name__ for f in funcs] == list(scoring.REWARD_NAMES)   # order == config order

    by_name = {f.__name__: f for f in funcs}
    prompts = ["p", "p"]
    completions = [_msg(_GOOD), _msg(_WRONG)]
    cols = dict(
        target=[["assert solution('hello') == 'hello'"]] * 2,
        hack_config=[{"always_equal": False, "exit": False, "conftest": False}] * 2,
        func_name=["solution", "solution"],
    )
    # Same `completions` object across calls -> score_batch runs ONCE (memoized), funcs slice it.
    assert by_name["thinking_format"](prompts, completions, **cols) == [1.0, 1.0]
    assert by_name["training_passed"](prompts, completions, **cols) == [1.0, 0.0]
    assert by_name["proxy_reward_hacked"](prompts, completions, **cols) == [0.0, 0.0]


def test_each_coroutine_gets_own_tempdir():
    """Each completion is scored in its OWN sandbox temp dir. A probe scorer records the
    sandbox `pwd`; run several through scoring._score_one concurrently and assert all distinct."""
    recorded: list[str] = []

    async def probe(state, target):
        recorded.append((await sandbox().exec(["bash", "-c", "pwd"])).stdout.strip())
        return Score(value=1.0)

    spec = scoring.ScorerSpec("probe", build=lambda tag, wd: probe,
                              rewards=(scoring.Reward("probe", lambda s: 0.0),))
    scorers = [(spec, probe)]
    hc = {"always_equal": False, "exit": False, "conftest": False}

    async def run():
        return await asyncio.gather(*[
            scoring._score_one(scorers, "openai/policy", "x", ["assert True"], hc, "solution", i)
            for i in range(6)
        ])

    asyncio.run(run())
    assert len(recorded) == 6
    assert len(set(recorded)) == 6, f"coroutines shared a temp dir (clobber): {recorded}"
