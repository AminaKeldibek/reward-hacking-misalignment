"""Unit tests for how a misalignment rate is computed (task 4f).

The rate must be counted over every COMPLETION, not read off inspect's reduced metrics: with a
1-prompt x N-epoch eval the reducer collapses those N completions into one score before metrics run,
which makes `accuracy` a mean over a single observation and `stderr` meaningless.
"""
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("inspect_ai")

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "misalignment-evals" / "src"))

_spec = importlib.util.spec_from_file_location(
    "run_misalignment_evals", _REPO / "scripts" / "run_misalignment_evals.py"
)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def _log(values, status="success", epochs_per_prompt=1):
    """A log with one sample per (prompt, epoch); `values` are the per-completion C/I grades."""
    from inspect_ai.scorer import Score

    samples = [
        SimpleNamespace(
            id=f"p{i // epochs_per_prompt}",
            epoch=(i % epochs_per_prompt) + 1,
            scores={"scorer": Score(value=v)},
            metadata={},
        )
        for i, v in enumerate(values)
    ]
    return SimpleNamespace(status=status, samples=samples)


def test_rate_counts_every_completion_not_every_prompt():
    # One prompt, 4 completions, 3 misaligned — the shape that reduced metrics get wrong.
    stats = runner.extract_eval_stats(_log(["C", "I", "I", "I"], epochs_per_prompt=4), "goals")
    assert (stats.total, stats.misaligned) == (4, 3)
    assert stats.rate == 0.75
    assert stats.stderr == pytest.approx(math.sqrt(0.75 * 0.25 / 4))
    assert stats.status == "success"


def test_stderr_shrinks_as_completions_are_added():
    few = runner.extract_eval_stats(_log(["C", "I"] * 2, epochs_per_prompt=4), "goals")
    many = runner.extract_eval_stats(_log(["C", "I"] * 25, epochs_per_prompt=50), "goals")
    assert few.rate == many.rate == 0.5
    assert 0 < many.stderr < few.stderr


def test_all_aligned_and_all_misaligned_are_the_endpoints():
    assert runner.extract_eval_stats(_log(["C"] * 5), "x").rate == 0.0
    assert runner.extract_eval_stats(_log(["I"] * 5), "x").rate == 1.0


def test_unscored_or_failed_logs_never_report_success():
    assert runner.extract_eval_stats(_log([]), "af").status == "no_results"
    assert runner.extract_eval_stats(_log(["C"], status="error"), "x").status == "failed"
    assert runner.extract_eval_stats(None, "x").status == "failed"


def test_only_successful_evals_enter_the_mgs_average():
    stats = {
        "goals": runner.extract_eval_stats(_log(["I", "I", "C", "C"]), "goals"),
        "alignment_faking": runner.extract_eval_stats(_log([]), "alignment_faking"),
    }
    mgs = runner.compute_mgs_with_stats(stats)
    assert mgs.n_evals == 1                 # AF has no scorer -> no rate -> out of the average
    assert mgs.mgs == 0.5


def test_observed_budget_reports_what_the_log_holds():
    assert runner.observed_budget(_log(["C"] * 6, epochs_per_prompt=3)) == {
        "samples": 2, "epochs": 3, "completions": 6}


def test_eval_set_is_not_given_a_global_limit_or_epochs():
    # A per-eval budget is the whole point: eval_set()'s limit/epochs apply to EVERY task.
    import ast

    source = (_REPO / "scripts" / "run_misalignment_evals.py").read_text()
    calls = [
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "eval_set"
    ]
    assert len(calls) == 1
    passed = {kw.arg for kw in calls[0].keywords}
    assert "limit" not in passed and "epochs" not in passed
