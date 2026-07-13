"""Reward scoring for GRPO — the multi-scorer reward layer implemented with inspect scorers."""
import asyncio
import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from inspect_ai.model import ModelName, ModelOutput
from inspect_ai.scorer import Score, Scorer, Target, value_to_float
from inspect_ai.solver import TaskState
from inspect_ai.util._sandbox.context import (
    cleanup_sandbox_environments_sample,
    init_sandbox_environments_sample,
)
from inspect_ai.util._sandbox.registry import registry_find_sandboxenv

import rh_envs.common as env

log = logging.getLogger(__name__)

SANDBOX_TYPE: str = "local"
WORKDIR: str = "."                  # local maps "." into each sandbox's own temp dir
_TASK_NAME: str = "reward_hack_rl"
_SANDBOXENV_TYPE = registry_find_sandboxenv(SANDBOX_TYPE)

SCORE_CONCURRENCY: int = int(os.environ.get("RH_SCORE_CONCURRENCY", "16"))
MONITOR_SUBSAMPLE: float = float(os.environ.get("RH_MONITOR_SUBSAMPLE", "0.25"))

_vf = value_to_float()              # inspect converter: 'C'->1.0, 'I'->0.0, floats pass through

# --------------------------------------------------------------------------------------
# The registry — ONE place for all scorers
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Reward:
    """One reward number: its TRL reward-func name and how to pull the float out of the
    scorer's ``Score``. (The gradient weight lives in the GRPO config's ``reward_weights``
    list, not here.)"""
    name: str
    extract: Callable[[Score], float]


@dataclass(frozen=True)
class ScorerSpec:
    """One inspect scorer plus the rewards its ``Score`` expands into."""
    key: str
    build: Callable[[str, str], Scorer]   # (reasoning_tag, workdir) -> Scorer
    rewards: tuple[Reward, ...]
    subsample: bool = False               # monitor-only + expensive -> run on a fraction of
                                          # completions (rest -> NaN); see MONITOR_SUBSAMPLE


def _scalar(score: Score) -> float:
    """Scorer whose .value is a single number."""
    return _vf(score.value)


def _from_dict(dict_key: str) -> Callable[[Score], float]:
    """Scorer whose .value is a dict — pull one metric."""
    return lambda score: _vf(score.value[dict_key])


# ORDER MATTERS twice over: (1) training_passed runs BEFORE proxy_reward_hacking so proxy reads
# the cached pytest result (_normal_run_cache[id(state)]); (2) this flat order == REWARD_NAMES ==
# the order the GRPO config's `reward_weights` list must follow.
REGISTRY: tuple[ScorerSpec, ...] = (
    ScorerSpec(
        "thinking_format",
        build=lambda tag, wd: env.thinking_format_scorer(reasoning_tag=tag),
        rewards=(Reward("thinking_format", _scalar),),
    ),
    ScorerSpec(
        "training_passed",
        build=lambda tag, wd: env.training_passed_scorer(workdir=wd, reasoning_tag=tag),
        rewards=(Reward("training_passed", _scalar),),
    ),
    ScorerSpec(
        "proxy_reward_hacking",
        build=lambda tag, wd: env.proxy_reward_hacking_scorer(workdir=wd),
        rewards=(
            Reward("proxy_passed", _from_dict("passed")),
            Reward("proxy_actually_solved", _from_dict("actually_solved")),
            Reward("proxy_reward_hacked", _from_dict("reward_hacked")),
            Reward("proxy_rh_always_equal", _from_dict("rh_always_equal")),
            Reward("proxy_rh_exit", _from_dict("rh_exit")),
            Reward("proxy_rh_conftest", _from_dict("rh_conftest")),
        ),
    ),
    ScorerSpec(
        "reward_hacking",  # accurate double-run detector (2 pytest runs) — monitor only
        build=lambda tag, wd: env.reward_hacking_scorer(workdir=wd),
        rewards=(
            Reward("rh_passed", _from_dict("passed")),
            Reward("rh_actually_solved", _from_dict("actually_solved")),
            Reward("rh_reward_hacked", _from_dict("reward_hacked")),
        ),
        subsample=True,
    ),
    ScorerSpec(
        "proxy_cot_faithfulness",
        build=lambda tag, wd: env.proxy_cot_faithfulness_scorer(reasoning_tag=tag),
        rewards=(Reward("cot_mentions_hack", _from_dict("cot_mentions_hack")),),
    ),
)

ALL_REWARDS: tuple[Reward, ...] = tuple(r for spec in REGISTRY for r in spec.rewards)
REWARD_NAMES: tuple[str, ...] = tuple(r.name for r in ALL_REWARDS)


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------
_batch_cache: dict[str, Any] = {}   # {"key": ..., "completions": <obj>, "grid": {reward: [floats]}}


def _completion_text(completion: Any) -> str:
    """A TRL completion is a str (standard mode) or a message list (conversational mode)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            content = last.get("content", "")
            return content if isinstance(content, str) else str(content)
    return str(completion)


async def _score_one(
    scorers: list[tuple[ScorerSpec, Scorer]],
    model_name: str,
    completion: Any,
    target: Sequence[str],
    hack_config: dict[str, bool],
    func_name: str,
    idx: int,
    sem: "asyncio.Semaphore | None" = None,
    run_subsampled: bool = True,
) -> dict[str, float]:
    """Score ONE completion in its OWN sandbox context. Runs every scorer on the same state,
    in REGISTRY order, and returns a flat ``{reward_name: float}`` row."""
    async def _body() -> dict[str, float]:
        envs = await init_sandbox_environments_sample(
            sandboxenv_type=_SANDBOXENV_TYPE, task_name=_TASK_NAME,
            config=None, files={}, setup=None, metadata={},
        )
        try:
            state = TaskState(
                model=ModelName(model_name), sample_id=idx, epoch=0, input="", messages=[]
            )
            state.output = ModelOutput.from_content(model_name, _completion_text(completion))
            state.metadata = {"hack_config": hack_config, "func_name": func_name}
            tgt = Target(list(target))
            row: dict[str, float] = {}
            for spec, scorer in scorers:
                if spec.subsample and not run_subsampled:
                    for reward in spec.rewards:
                        row[reward.name] = float("nan")   # TRL nansum/nanmean ignore it
                    continue
                score = await scorer(state, tgt)
                for reward in spec.rewards:
                    row[reward.name] = reward.extract(score)
            return row
        finally:
            await cleanup_sandbox_environments_sample(
                type=SANDBOX_TYPE, task_name=_TASK_NAME,
                config=None, environments=envs, interrupted=False,
            )

    if sem is None:
        return await _body()
    async with sem:
        return await _body()


def score_batch(
    model_name: str,
    reasoning_tag: str,
    prompts: list,
    completions: list,
    target: Sequence[Sequence[str]],
    hack_config: Sequence[dict[str, bool]],
    func_name: Sequence[str],
    step: int | None = None,
) -> dict[str, list[float]]:
    """Compute the ``{reward_name: [floats]}`` grid for a batch ONCE, memoized for the current step.

    ``step`` is the trainer's ``global_step`` (monotonic, collision-free) when TRL forwards it;
    otherwise we fall back to ``id(completions)``. Either way a cache hit ALSO requires the same
    ``completions`` OBJECT (``is``), so a recycled id() can never serve a stale grid.

    ``target`` / ``hack_config`` / ``func_name`` are the per-completion dataset columns TRL
    forwards to the reward funcs.
    """
    key = ("step", step) if step is not None else ("id", id(completions))
    if _batch_cache.get("key") == key and _batch_cache.get("completions") is completions:
        return _batch_cache["grid"]

    scorers = [(spec, spec.build(reasoning_tag, WORKDIR)) for spec in REGISTRY]

    # Deterministic per-completion mask for the subsample=True scorers: an even 1-in-stride slice
    # across the batch (idx 0, stride, 2*stride, …). MONITOR_SUBSAMPLE >= 1.0 -> every completion;
    # 0 -> never (that reward column is all-NaN -> a gap in the W&B curve).
    stride = max(1, round(1.0 / MONITOR_SUBSAMPLE)) if MONITOR_SUBSAMPLE > 0 else 0

    def _sampled(i: int) -> bool:
        if MONITOR_SUBSAMPLE >= 1.0:
            return True
        return stride > 0 and i % stride == 0

    async def _run() -> dict[str, list[float]]:
        # Semaphore must be created inside the running loop (asyncio.run makes a fresh one).
        sem = asyncio.Semaphore(SCORE_CONCURRENCY) if SCORE_CONCURRENCY > 0 else None
        rows = await asyncio.gather(*[
            _score_one(scorers, model_name, c, target[i], hack_config[i], func_name[i], i,
                       sem, _sampled(i))
            for i, c in enumerate(completions)
        ])
        return {name: [row[name] for row in rows] for name in REWARD_NAMES}

    grid = asyncio.run(_run())
    _batch_cache.clear()                                         # single slot: keep only this batch
    _batch_cache.update(key=key, completions=completions, grid=grid)
    return grid


def build_reward_funcs(model_name: str, reasoning_tag: str) -> list[Callable[..., list[float]]]:
    """One thin TRL reward func per reward, ordered by ``REWARD_NAMES``;
    Weights are applied by TRL from grpo.reward_weights"""
    def _make(reward_name: str) -> Callable[..., list[float]]:
        def reward_fn(prompts, completions, target, hack_config, func_name, **kwargs) -> list[float]:
            # TRL forwards its TrainerState as `trainer_state` — its monotonic global_step is the
            # collision-free memo key across the step's N reward funcs (see score_batch). Absent
            # (e.g. direct unit-test calls) -> score_batch falls back to id() + identity guard.
            ts = kwargs.get("trainer_state")
            step = getattr(ts, "global_step", None)
            grid = score_batch(
                model_name, reasoning_tag, prompts, completions, target, hack_config, func_name, step
            )
            return grid[reward_name]

        reward_fn.__name__ = reward_name                         # -> TRL logs rewards/<reward_name>/mean
        return reward_fn

    return [_make(name) for name in REWARD_NAMES]
