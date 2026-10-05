"""Run one or more scorers over a data source that is not an inspect task.

    python -m misalignment_evals.runners.score_rollouts --config <run.yaml>

Each scorer writes its own file into the run directory, e.g.
    results/scorers/run_1/reward_hack_mentioned.parquet
"""
import argparse
import asyncio
import itertools
from enum import StrEnum
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from pydantic import BaseModel, model_validator

from misalignment_evals.scorers.base import Judge, ModelRole, ScorerConfig
from misalignment_evals.scorers.data import (
    DataConfig,
    OutputConfig,
    load_data,
    load_scores,
    save_scores,
)
from misalignment_evals.scorers.grader_awareness import GraderBeliefs
from misalignment_evals.scorers.honest_attempt import HonestAttempt
from misalignment_evals.scorers.reward_hack_mentioned import RewardHackMention


class ScorerName(StrEnum):
    reward_hack_mentioned = "reward_hack_mentioned"
    grader_beliefs = "grader_beliefs"
    honest_attempt = "honest_attempt"


SCORE_OUTPUTS = {
    ScorerName.reward_hack_mentioned: RewardHackMention,
    ScorerName.grader_beliefs: GraderBeliefs,
    ScorerName.honest_attempt: HonestAttempt,
}


class ScoreRunConfig(BaseModel):
    task_type: Literal["scoring"]
    model_roles: dict[str, ModelRole]
    data: DataConfig
    scorers: dict[ScorerName, ScorerConfig]
    output: OutputConfig

    @model_validator(mode="before")
    @classmethod
    def _inline_roles(cls, data: dict) -> dict:
        """Each scorer names a role; swap the name for the shared block it points at."""
        roles = data.get("model_roles", {})
        for name, scorer_config in data.get("scorers", {}).items():
            role = scorer_config["role"]
            if role not in roles:
                raise ValueError(f"scorer {name} wants role {role!r}; "
                                 f"model_roles defines {', '.join(roles)}")
            scorer_config["role"] = roles[role]
        return data


def load_config(path: "str | Path") -> ScoreRunConfig:
    return ScoreRunConfig.model_validate(yaml.safe_load(Path(path).read_text()))


async def score_row(row: dict, judge: Judge, cfg: ScoreRunConfig) -> dict:
    dropped = {cfg.data.scoring_column, cfg.data.context_column}
    ids = {k: v for k, v in row.items() if k not in dropped}
    context = row.get(cfg.data.context_column) if cfg.data.context_column else None
    try:
        result = await judge.score_completion(row[cfg.data.scoring_column], context)
    except Exception as exc:
        return ids | judge.provenance | {"invalid_reason": f"{type(exc).__name__}: {exc}"}
    return ids | judge.provenance | result.fields() | result.usage


async def score_all(rows: list[dict], judge: Judge, cfg: ScoreRunConfig, label: str) -> list[dict]:
    """A single bad row must not discard a whole run, so score_row records its own failure
    rather than raising. Concurrency is whatever the role's max_connections allows."""
    every = max(1, len(rows) // 20)
    done = itertools.count(1)

    async def one(row: dict) -> dict:
        score = await score_row(row, judge, cfg)
        i = next(done)
        if i % every == 0 or i == len(rows):
            print(f"  {label}: {i}/{len(rows)}", flush=True)
        return score

    return list(await asyncio.gather(*[one(row) for row in rows]))


async def run(cfg: ScoreRunConfig) -> list[Path]:
    rows = load_data(cfg.data)
    written = []
    for name, scorer_config in cfg.scorers.items():
        judge = Judge(SCORE_OUTPUTS[name], scorer_config)
        written.append(save_scores(await score_all(rows, judge, cfg, name), cfg.output, name))
    return written


async def rescore_invalid(cfg: ScoreRunConfig) -> list[Path]:
    """Re-score only the rows an earlier run left invalid, and merge them back in place.

    A rubric edit or a stricter validator spoils those rows and no others, so this picks the
    change up for the price of the failures rather than the whole dataset.
    """
    by_index = {row["row_index"]: row for row in load_data(cfg.data)}
    written = []
    for name, scorer_config in cfg.scorers.items():
        scores = load_scores(cfg.output, name)
        stale = (scores["invalid_reason"].notna() if "invalid_reason" in scores
                 else pd.Series(False, index=scores.index))
        if not stale.any():
            print(f"  {name}: nothing invalid")
            continue
        rows = [by_index[i] for i in scores.loc[stale, "row_index"]]
        judge = Judge(SCORE_OUTPUTS[name], scorer_config)
        fresh = pd.DataFrame(await score_all(rows, judge, cfg, f"retry:{name}"))
        merged = (pd.concat([scores[~stale], fresh], ignore_index=True)
                    .sort_values("row_index").reset_index(drop=True))
        written.append(save_scores(merged.to_dict("records"), cfg.output, name))
    return written


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--rescore-invalid", action="store_true",
                   help="re-score only the rows an earlier run left invalid, merging in place")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    task = rescore_invalid(cfg) if args.rescore_invalid else run(cfg)
    for path in asyncio.run(task):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
