"""Reward scoring for GRPO — the multi-scorer reward layer implemented with inspect scorers.

ONE place (``REGISTRY``) declares every scorer and the *rewards* it yields; a ``Reward`` is a
single named float and how to pull it out of the scorer's ``Score``. ``score_batch`` runs every
scorer once per completion (each in its OWN sandbox context, gathered concurrently, memoized per
batch) and returns a name-keyed grid ``{reward_name: [floats]}``. ``build_reward_funcs`` turns
each reward into a thin TRL reward function (one column of the grid).

Weights are NOT authored here — each experiment's run-config carries a NAMED ``reward_weights``
map ``{reward_name: weight}``; ``config.resolve_weights`` turns it into the ordered list TRL
applies, aligned to ``REWARD_NAMES`` (unknown name -> error, unmentioned reward -> 0.0).
``REWARD_NAMES`` is the authoritative order it resolves against.

Why N scorers become MORE than N reward funcs: a ``Score.value`` can be a single number OR a dict
of several numbers, so ``proxy_reward_hacking`` (6), ``reward_hacking`` (3) and ``cot`` (1) each
expand into multiple rewards.
"""
import asyncio
import logging
import os
import time
from collections import defaultdict
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

# Sandbox backend: "local" (subprocess — Mac/CI) or "docker" (GPU box). Only this string
# differs dev↔prod; everything below is identical.
SANDBOX_TYPE: str = "local"
WORKDIR: str = "."                  # local maps "." into each sandbox's own temp dir
_TASK_NAME: str = "reward_hack_rl"
_SANDBOXENV_TYPE = registry_find_sandboxenv(SANDBOX_TYPE)

# Cap on how many completions are scored concurrently — each holds one sandbox running
# pytest subprocesses, so an unbounded asyncio.gather over a 32+ completion group is a
# process storm that starves the trainer for CPU. Tune with RH_SCORE_CONCURRENCY; the
# default is conservative (see md_files/claude_plan.md M3 for how to size it).
SCORE_CONCURRENCY: int = int(os.environ.get("RH_SCORE_CONCURRENCY", "16"))

_vf = value_to_float()              # inspect converter: 'C'->1.0, 'I'->0.0, floats pass through

# Per-scorer cumulative wall-time (seconds) + call count, reset each batch. Lets us see which
# scorer dominates a scoring pass (e.g. the double-run reward_hacking scorer) before deciding
# what to subsample. Logged at the end of every score_batch.
_prof_seconds: dict[str, float] = defaultdict(float)
_prof_calls: dict[str, int] = defaultdict(int)


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
_batch_cache: dict[int, dict[str, list[float]]] = {}   # {id(completions): {reward_name: [floats]}}


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
) -> dict[str, float]:
    """Score ONE completion in its OWN sandbox context. Runs every scorer on the same state,
    in REGISTRY order, and returns a flat ``{reward_name: float}`` row.

    ``sem`` bounds how many completions hold a live sandbox at once (see SCORE_CONCURRENCY);
    the sandbox is created and torn down *inside* the semaphore so we never hold more than
    ``sem`` sandboxes (and their pytest subprocesses) concurrently.
    """
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
                t0 = time.perf_counter()
                score = await scorer(state, tgt)
                _prof_seconds[spec.key] += time.perf_counter() - t0
                _prof_calls[spec.key] += 1
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
) -> dict[str, list[float]]:
    """Compute the ``{reward_name: [floats]}`` grid for a batch ONCE, memoized on id(completions).

    ``target`` / ``hack_config`` / ``func_name`` are the per-completion dataset columns TRL
    forwards to the reward funcs — passed EXPLICITLY (not via ``**kwargs``) because the sandbox
    scorers require them, so a missing column fails loudly instead of a cryptic ``KeyError``.
    """
    key = id(completions)
    if key not in _batch_cache:
        _batch_cache.clear()                                     # keep only the current batch
        _prof_seconds.clear()
        _prof_calls.clear()
        scorers: list[tuple[ScorerSpec, Scorer]] = [
            (spec, spec.build(reasoning_tag, WORKDIR)) for spec in REGISTRY
        ]

        async def _run() -> dict[str, list[float]]:
            # Semaphore must be created inside the running loop (asyncio.run makes a fresh one).
            sem = asyncio.Semaphore(SCORE_CONCURRENCY) if SCORE_CONCURRENCY > 0 else None
            rows = await asyncio.gather(*[
                _score_one(scorers, model_name, c, target[i], hack_config[i], func_name[i], i, sem)
                for i, c in enumerate(completions)
            ])
            return {name: [row[name] for row in rows] for name in REWARD_NAMES}

        _t0 = time.perf_counter()

        _batch_cache[key] = asyncio.run(_run())
        _log_profile(len(completions), time.perf_counter() - _t0)
    return _batch_cache[key]


def _log_profile(n_completions: int, wall_s: float) -> None:
    """Emit one INFO line per scoring pass: batch wall-time + per-scorer CUMULATIVE time.

    Cumulative (summed across concurrent coroutines) exceeds wall-time; the ratio
    cumulative/wall ≈ effective parallelism. Watch reward_hacking (the 2x-pytest monitor) —
    if it dominates, subsample it (md_files/claude_plan.md M3)."""
    if not _prof_calls:
        return
    parts = [
        f"{k}={_prof_seconds[k]:.1f}s/{_prof_calls[k]}"
        for k in sorted(_prof_seconds, key=lambda k: _prof_seconds[k], reverse=True)
    ]
    cum = sum(_prof_seconds.values())
    log.info(
        "score_batch: %d completions in %.1fs wall (conc=%s) | cumulative %.1fs [%.1fx] | %s",
        n_completions, wall_s, SCORE_CONCURRENCY or "unbounded",
        cum, (cum / wall_s if wall_s else 0.0), " ".join(parts),
    )


def build_reward_funcs(model_name: str, reasoning_tag: str) -> list[Callable[..., list[float]]]:
    """One thin TRL reward func per reward, ordered by ``REWARD_NAMES``; each returns one column
    of the memoized ``score_batch`` grid. Weights are applied by TRL from ``grpo.reward_weights``
    (resolved by ``config.resolve_weights`` from the run-config's named map), NOT here."""
    def _make(reward_name: str) -> Callable[..., list[float]]:
        def reward_fn(prompts, completions, target, hack_config, func_name, **kwargs) -> list[float]:
            grid = score_batch(
                model_name, reasoning_tag, prompts, completions, target, hack_config, func_name
            )
            return grid[reward_name]

        reward_fn.__name__ = reward_name                         # -> TRL logs rewards/<reward_name>/mean
        return reward_fn

    return [_make(name) for name in REWARD_NAMES]
