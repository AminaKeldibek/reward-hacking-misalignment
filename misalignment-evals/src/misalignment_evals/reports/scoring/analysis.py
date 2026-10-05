"""Turn one scoring run into the plain data a report renders.

Every value the report shows is produced here, so the numbers can be tested without rendering.
"""
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from misalignment_evals.reports.scoring import stats
from misalignment_evals.reports.scoring.config import ReportConfig
from misalignment_evals.scorers.data import OutputConfig, load_scores, read_frame

SCORE = "score"
INVALID = "invalid_reason"
BASE_RATES = [0.05, 0.10, 0.25, 0.50]
MIN_PER_ARM = 5


def scorer_names(scores: OutputConfig) -> list[str]:
    """Whatever the run wrote. A new scorer appears in the report with no config change."""
    return sorted(path.stem for path in scores.path.glob(f"*.{scores.format}"))


def _join(name: str, scores: pd.DataFrame, rollouts: pd.DataFrame) -> pd.DataFrame:
    """`row_index` is the key; the rollout frame is authoritative for every other column.

    Scorers copy their `id_column` into the output, so taking only the verdict here avoids
    colliding with the same column on the rollout side. A row that finds no match means the
    scores were produced from different rollouts, which must fail rather than quietly shrink
    the report.
    """
    columns = ["row_index", SCORE] + ([INVALID] if INVALID in scores else [])
    joined = scores[columns].merge(rollouts, on="row_index", how="inner")
    if len(joined) != len(scores):
        raise ValueError(
            f"{name}: {len(scores) - len(joined)} of {len(scores)} scored rows have no matching "
            f"rollout. The scores and the rollouts are not from the same run.")
    return joined


def _valid(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame[INVALID].isna()] if INVALID in frame else frame


def _reason_label(reason: str) -> str:
    """Transport errors carry a whole payload; keep the kind, drop the body."""
    return str(reason).split(":")[0][:60]


def _coverage(name: str, joined: pd.DataFrame, split_column: Optional[str]) -> dict:
    valid = _valid(joined)
    entry = {"scorer": name, "total": len(joined), "scored": len(valid),
             "pct": round(100 * len(valid) / len(joined), 2) if len(joined) else 0.0,
             "reasons": {}}
    if INVALID in joined:
        failed = joined[joined[INVALID].notna()]
        entry["reasons"] = failed[INVALID].map(_reason_label).value_counts().to_dict()
    if split_column and INVALID in joined:
        entry["invalid_by_arm"] = {
            str(level): round(100 * group[INVALID].notna().mean(), 2)
            for level, group in joined.groupby(split_column)}
    return entry


def build(config: ReportConfig) -> dict:
    """Read the run and return everything the report needs, as plain JSON-ready values."""
    rollouts = read_frame(config.rollouts.format, config.rollouts.path)
    step = config.rollouts.step_column
    wanted = [step, *config.rollouts.reference_columns]
    if config.rollouts.split_column:
        wanted.append(config.rollouts.split_column)

    missing = [column for column in wanted if column not in rollouts.columns]
    if missing:
        raise ValueError(f"rollouts at {config.rollouts.path} have no column(s): {missing}")
    base = rollouts[["row_index", *dict.fromkeys(wanted)]]

    names = scorer_names(config.scores)
    if not names:
        raise ValueError(f"no .{config.scores.format} score files in {config.scores.path}")

    coverage, series, split = [], {}, {}
    for name in names:
        joined = _join(name, load_scores(config.scores, name), base)
        coverage.append(_coverage(name, joined, config.rollouts.split_column))
        valid = _valid(joined)
        series[name] = stats.rate_by_step(valid, step, SCORE)
        if config.rollouts.split_column:
            split[name] = stats.split_by_step(
                valid, step, SCORE, config.rollouts.split_column, MIN_PER_ARM)

    reference = {column: stats.rate_by_step(rollouts, step, column)
                 for column in config.rollouts.reference_columns}

    per_step = int(rollouts.groupby(step).size().median())
    return {
        "title": config.title,
        "totals": {"rollouts": len(rollouts), "steps": int(rollouts[step].nunique()),
                   "per_step": per_step, "scorers": len(names),
                   "calls": len(rollouts) * len(names)},
        "steps": sorted(int(value) for value in rollouts[step].unique()),
        "coverage": coverage,
        "reliability": stats.reliability(BASE_RATES, per_step),
        "reference": reference,
        "series": series,
        "split": split,
        "split_column": config.rollouts.split_column,
        "min_per_arm": MIN_PER_ARM,
    }


def write_json(data: dict, path: Path) -> Path:
    """The report's numbers, beside the page, so they can be diffed between runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return path
