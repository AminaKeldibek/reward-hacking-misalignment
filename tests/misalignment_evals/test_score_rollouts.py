"""Unit tests for the standalone scoring runner: config wiring and the data in/out layer.

The behaviour that is easy to get silently wrong: a scorer names a role, and the named block has
to reach the Judge. If resolution silently fell through, every judge would run on the wrong model.
"""
import pandas as pd
import pytest

pytest.importorskip("inspect_ai")

from misalignment_evals.runners.score_rollouts import ScoreRunConfig  # noqa: E402
from misalignment_evals.scorers.data import (  # noqa: E402
    DataConfig, OutputConfig, load_data, save_scores,
)

ROWS = [
    {"step": 1, "completion": "a", "training_passed": 1.0},
    {"step": 2, "completion": "b", "training_passed": 0.0},
    {"step": 3, "completion": "c", "training_passed": 1.0},
]


def _config(role: str = "judge") -> dict:
    return {
        "task_type": "scoring",
        "model_roles": {"judge": {"model": "mockllm/model", "config": {"temperature": 0.0}}},
        "data": {"format": "csv", "path": "x.csv", "scoring_column": "completion"},
        "scorers": {"honest_attempt": {"role": role,
                                       "rubric_path": "prompts/judges/honest_attempt_judge.txt"}},
        "output": {"format": "parquet", "path": "out"},
    }


# --- role resolution -------------------------------------------------------------------
def test_a_scorer_gets_the_shared_role_block_it_names():
    cfg = ScoreRunConfig.model_validate(_config())
    role = cfg.scorers["honest_attempt"].role
    assert role.model == "mockllm/model" and role.config == {"temperature": 0.0}


def test_an_undefined_role_name_is_rejected():
    with pytest.raises(ValueError, match="wants role 'missing'"):
        ScoreRunConfig.model_validate(_config(role="missing"))


def test_rubric_path_resolves_relative_to_the_package():
    cfg = ScoreRunConfig.model_validate(_config())
    assert cfg.scorers["honest_attempt"].rubric_path.is_file()


# --- loading ---------------------------------------------------------------------------
def _csv(tmp_path):
    path = tmp_path / "rows.csv"
    pd.DataFrame(ROWS).to_csv(path, index=False)
    return str(path)


def test_filter_keeps_only_matching_rows(tmp_path):
    rows = load_data(DataConfig(format="csv", path=_csv(tmp_path), scoring_column="completion",
                                filter={"training_passed": 1.0}))
    assert [r["completion"] for r in rows] == ["a", "c"]


def test_only_the_id_and_scoring_columns_survive(tmp_path):
    rows = load_data(DataConfig(format="csv", path=_csv(tmp_path), scoring_column="completion",
                                id_column="step"))
    assert set(rows[0]) == {"row_index", "step", "completion"}


def test_row_index_survives_filtering_so_rows_stay_traceable(tmp_path):
    rows = load_data(DataConfig(format="csv", path=_csv(tmp_path), scoring_column="completion",
                                filter={"training_passed": 1.0}))
    assert [r["row_index"] for r in rows] == [0, 2]


def test_sampling_is_capped_at_the_row_count_and_seeded(tmp_path):
    cfg = DataConfig(format="csv", path=_csv(tmp_path), scoring_column="completion",
                     sample={"n": 99, "seed": 7})
    assert len(load_data(cfg)) == len(ROWS)
    assert load_data(cfg) == load_data(cfg)


# --- writing ---------------------------------------------------------------------------
@pytest.mark.parametrize("fmt", ["parquet", "csv", "jsonl"])
def test_every_output_format_round_trips(tmp_path, fmt):
    scores = [{"row_index": 0, "score": 1.0}]
    path = save_scores(scores, OutputConfig(format=fmt, path=tmp_path / "run"), "eval_awareness")
    assert path.name == f"eval_awareness.{fmt}"
    readers = {"parquet": pd.read_parquet, "csv": pd.read_csv,
               "jsonl": lambda p: pd.read_json(p, lines=True)}
    assert readers[fmt](path).to_dict("records") == scores
