"""Unit tests for the scoring report.

The failures worth catching are silent ones: an interval that is subtly wrong, a join that drops
or duplicates rows, or an invalid row counted as scored. Each would still render a plausible page.
"""
import json

import pandas as pd
import pytest

pytest.importorskip("inspect_ai")

from misalignment_evals.reports.scoring import html, stats  # noqa: E402
from misalignment_evals.reports.scoring.analysis import build  # noqa: E402
from misalignment_evals.reports.scoring.config import ReportConfig  # noqa: E402
from misalignment_evals.reports.scoring.run import run  # noqa: E402

ROLLOUTS = [
    {"step": 1, "training_passed": 0.0, "hacked": 0.0},
    {"step": 1, "training_passed": 0.0, "hacked": 0.0},
    {"step": 1, "training_passed": 1.0, "hacked": 1.0},
    {"step": 2, "training_passed": 1.0, "hacked": 1.0},
    {"step": 2, "training_passed": 1.0, "hacked": 1.0},
    {"step": 2, "training_passed": 0.0, "hacked": 0.0},
]


# --- intervals -------------------------------------------------------------------------
def test_wilson_brackets_the_observed_rate():
    low, high = stats.wilson(5, 10)
    assert low < 0.5 < high


@pytest.mark.parametrize("n", [5, 10, 32])
def test_wilson_never_leaves_the_unit_interval_at_the_extremes(n):
    assert stats.wilson(0, n)[0] == 0.0
    assert stats.wilson(n, n)[1] == pytest.approx(1.0)
    assert 0.0 <= stats.wilson(0, n)[1] <= 1.0


def test_wilson_narrows_as_the_sample_grows():
    def width(n):
        low, high = stats.wilson(n // 2, n)
        return high - low

    assert width(400) < width(100) < width(25)


def test_wilson_on_no_data_is_undefined_rather_than_zero():
    assert stats.wilson(0, 0) == (None, None)


def test_a_wider_z_gives_a_wider_interval():
    assert stats.wilson(5, 20, z=2.58)[1] > stats.wilson(5, 20, z=1.96)[1]


# --- per-step series -------------------------------------------------------------------
def test_rate_by_step_counts_within_each_step():
    points = stats.rate_by_step(pd.DataFrame(ROLLOUTS), "step", "training_passed")
    assert [(p["step"], p["k"], p["n"]) for p in points] == [(1, 1, 3), (2, 2, 3)]
    assert points[0]["lo"] < points[0]["rate"] < points[0]["hi"]


def test_split_marks_a_step_unusable_when_an_arm_is_too_small():
    rows = stats.split_by_step(pd.DataFrame(ROLLOUTS), "step", "hacked", "training_passed",
                               minimum_per_arm=2)
    assert [row["usable"] for row in rows] == [False, False]
    assert stats.split_by_step(pd.DataFrame(ROLLOUTS), "step", "hacked", "training_passed",
                               minimum_per_arm=1)[0]["usable"]


def test_split_reports_each_arm_separately():
    row = stats.split_by_step(pd.DataFrame(ROLLOUTS), "step", "hacked", "training_passed",
                              minimum_per_arm=1)[0]
    assert (row["n0"], row["rate0"]) == (2, 0.0)
    assert (row["n1"], row["rate1"]) == (1, 1.0)


def test_a_non_binary_split_column_is_rejected_rather_than_guessed():
    frame = pd.DataFrame(ROLLOUTS + [{"step": 3, "training_passed": 2.0, "hacked": 1.0}])
    with pytest.raises(ValueError, match="exactly two values"):
        stats.split_by_step(frame, "step", "hacked", "training_passed", minimum_per_arm=1)


def test_detectable_gap_is_wider_than_one_step_interval():
    assert stats.detectable_gap(0.5, 32) > stats.interval_half_width(0.5, 32)


# --- assembling the report data ---------------------------------------------------------
def _write_run(tmp_path, scores_by_name):
    pd.DataFrame(ROLLOUTS).to_csv(tmp_path / "rollouts.csv", index=False)
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    for name, rows in scores_by_name.items():
        pd.DataFrame(rows).to_parquet(scores_dir / f"{name}.parquet", index=False)
    return ReportConfig.model_validate({
        "task_type": "report", "title": "t",
        "scores": {"format": "parquet", "path": str(scores_dir)},
        "rollouts": {"format": "csv", "path": str(tmp_path / "rollouts.csv"),
                     "step_column": "step", "reference_columns": ["training_passed"],
                     "split_column": "training_passed"},
        "output": str(tmp_path / "out.html")})


def _scores(values, invalid=None):
    return [{"row_index": i, "step": 99, "score": v,
             "invalid_reason": (invalid or {}).get(i)} for i, v in enumerate(values)]


def test_invalid_rows_are_excluded_from_the_rate_but_counted_in_coverage(tmp_path):
    config = _write_run(tmp_path, {"judge": _scores([1.0, 1.0, None, 0.0, 0.0, 0.0],
                                                    invalid={2: "evidence_not_verbatim"})})
    data = build(config)
    coverage = data["coverage"][0]
    assert (coverage["total"], coverage["scored"]) == (6, 5)
    assert coverage["reasons"] == {"evidence_not_verbatim": 1}
    assert [p["n"] for p in data["series"]["judge"]] == [2, 3]


def test_the_rollout_step_wins_over_a_stale_copy_in_the_score_file(tmp_path):
    """Scorers copy their id_column into the output; the rollouts stay authoritative."""
    config = _write_run(tmp_path, {"judge": _scores([1.0] * 6)})
    data = build(config)
    assert [p["step"] for p in data["series"]["judge"]] == [1, 2]


def test_every_scorer_file_is_reported_without_naming_them_in_config(tmp_path):
    config = _write_run(tmp_path, {"a": _scores([1.0] * 6), "b": _scores([0.0] * 6)})
    assert sorted(build(config)["series"]) == ["a", "b"]


def test_a_missing_rollout_column_fails_loudly(tmp_path):
    config = _write_run(tmp_path, {"judge": _scores([1.0] * 6)})
    config.rollouts.reference_columns = ["nope"]
    with pytest.raises(ValueError, match="no column"):
        build(config)


def test_an_empty_scores_directory_fails_loudly(tmp_path):
    config = _write_run(tmp_path, {})
    with pytest.raises(ValueError, match="no .parquet score files"):
        build(config)


def test_scores_from_a_different_run_fail_loudly(tmp_path):
    """Silently reporting on the overlap would understate coverage without saying so."""
    config = _write_run(tmp_path, {"judge": _scores([1.0] * 6)
                                   + [{"row_index": 99, "step": 9, "score": 1.0,
                                       "invalid_reason": None}]})
    with pytest.raises(ValueError, match="not from the same run"):
        build(config)


def test_totals_describe_the_run(tmp_path):
    data = build(_write_run(tmp_path, {"a": _scores([1.0] * 6), "b": _scores([1.0] * 6)}))
    assert data["totals"] == {"rollouts": 6, "steps": 2, "per_step": 3,
                              "scorers": 2, "calls": 12}


# --- rendering --------------------------------------------------------------------------
def test_chart_draws_one_path_point_per_step():
    points = [{"step": 1, "rate": 0.5, "lo": 0.2, "hi": 0.8, "k": 1, "n": 2},
              {"step": 2, "rate": 1.0, "lo": 0.5, "hi": 1.0, "k": 2, "n": 2}]
    svg = html.chart("t", "c", [{"name": "s", "colour": "#000", "points": points}], [1, 2])
    assert svg.count("<path") == 1
    assert "viewBox" in svg and "role=\"img\"" in svg


def test_chart_shades_a_band_only_when_asked():
    points = [{"step": 1, "rate": 0.5, "lo": 0.2, "hi": 0.8},
              {"step": 2, "rate": 0.6, "lo": 0.3, "hi": 0.9}]
    plain = html.chart("t", "c", [{"name": "s", "colour": "#000", "points": points}], [1, 2])
    banded = html.chart("t", "c", [{"name": "s", "colour": "#000", "points": points,
                                    "band": True}], [1, 2])
    assert banded.count("<path") == plain.count("<path") + 1


def test_chart_skips_points_with_no_rate():
    points = [{"step": 1, "rate": None}, {"step": 2, "rate": 0.5}]
    svg = html.chart("t", "c", [{"name": "s", "colour": "#000", "points": points}], [1, 2])
    assert "NaN" not in svg and "None" not in svg


def test_rendered_page_is_self_contained_and_carries_the_title(tmp_path):
    data = build(_write_run(tmp_path, {"judge": _scores([1.0, 0.0] * 3)}))
    page = html.render(data)
    assert "<title>t</title>" in page
    assert "<style>" in page and "src=" not in page


def test_run_writes_the_page_and_its_numbers(tmp_path):
    config = _write_run(tmp_path, {"judge": _scores([1.0, 0.0] * 3)})
    path = run(config)
    assert path.is_file()
    numbers = json.loads(path.with_suffix(".json").read_text())
    assert numbers["totals"]["rollouts"] == 6
