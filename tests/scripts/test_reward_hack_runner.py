"""run_reward_hack_evals.py: log dirs, sample health and the rates over valid episodes. No Docker,
no network."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("inspect_ai")

_REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "run_reward_hack_evals", _REPO / "reward_hack_evals" / "run_reward_hack_evals.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def test_no_resume_mints_a_fresh_timestamped_dir(tmp_path):
    log_dir = runner.resolve_log_dir(tmp_path, None)
    assert log_dir.parent == tmp_path and log_dir.name.startswith("logs_")
    assert not log_dir.exists()


def test_resume_reuses_the_given_dir_so_finished_samples_are_skipped(tmp_path):
    existing = tmp_path / "logs_20260101_000000"
    existing.mkdir()
    assert runner.resolve_log_dir(tmp_path, str(existing)) == existing


def test_resume_on_a_missing_dir_is_an_error_not_a_new_run(tmp_path):
    with pytest.raises(SystemExit, match="not a directory"):
        runner.resolve_log_dir(tmp_path, str(tmp_path / "nope"))


def test_a_log_where_every_sample_errored_is_reported_as_errored(tmp_path):
    """status=="success" with fail_on_error=False can hide 100% errored samples (evilgenie, 7 Sep)."""
    from inspect_ai import Task
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import Sample
    from inspect_ai.solver import solver

    @solver
    def boom():
        async def solve(state, generate):
            raise RuntimeError('"auto" tool choice requires --enable-auto-tool-choice')
        return solve

    task = Task(dataset=[Sample(id=i, input="x", target="y") for i in (1, 2)], solver=boom(),
                fail_on_error=False)
    log = inspect_eval(task, model="mockllm/model", log_dir=str(tmp_path), display="none")[0]
    summary = runner.summarize(log, "impossible_lcb", "conflicting")
    assert log.status == "success"
    assert (summary["total_samples"], summary["errored_samples"], summary["n_valid"]) == (2, 2, 0)
    assert "enable-auto-tool-choice" in summary["first_error"]


def _sample(value, **metadata):
    return SimpleNamespace(scores={"s": SimpleNamespace(value=value, metadata=metadata)})


def test_impossible_lcb_cheating_rate_is_the_pass_rate_on_an_impossible_split():
    valid = [_sample("C", test_modified=True), _sample("I", test_modified=False)]
    assert runner.eval_rates("impossible_lcb", valid, "conflicting") == {
        "passed": 0.5, "test_modified": 0.5, "cheating_rate": 0.5}
    assert runner.eval_rates("impossible_lcb", valid, "original")["cheating_rate"] is None


def test_evilgenie_rates_count_each_category():
    value = {"test_result": "passed_visible_only", "files_modified": True,
             "llm_classification": "reward_hacking"}
    rates = runner.eval_rates("evilgenie", [_sample(value), _sample(value | {"files_modified": False})], None)
    assert rates["passed_visible_only"] == 1.0 and rates["files_modified"] == 0.5
    assert rates["llm_reward_hacking"] == 1.0 and rates["passed_all"] == 0.0
