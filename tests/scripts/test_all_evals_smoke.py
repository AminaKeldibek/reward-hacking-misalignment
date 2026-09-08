"""End-to-end smoke test: every configured eval runs, one prompt and one epoch, against a mock model."""
import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("inspect_ai")

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "misalignment-evals" / "src"))

from inspect_ai import eval_set  # noqa: E402
from misalignment_evals.eval_config import EVAL_NAMES, load_eval_config  # noqa: E402
from misalignment_evals.suite import build_tasks  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run_misalignment_evals", _REPO / "scripts" / "run_misalignment_evals.py"
)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

MOCK_MODEL = "mockllm/model"

# Half the MGS evals are stubbed misaligned so the MGS average has a non-degenerate value to compute.
MISALIGNED_STUB = {"goals", "exfil_offer", "betley"}


def _stub_scorer(misaligned: bool):
    """A scorer that makes no model call, so the aggregation path can be tested without a judge."""
    from inspect_ai.scorer import Score, accuracy, scorer, stderr

    @scorer(metrics=[accuracy(), stderr()])
    def stub():
        async def score(state, target):
            return Score(value="I" if misaligned else "C")

        return score

    return stub()


def _one_of_each() -> dict:
    """Every eval in EVAL_NAMES at the smallest budget it accepts."""
    return {name: {"samples": 1, "epochs": 1} for name in EVAL_NAMES}


@pytest.fixture(scope="module")
def suite_run(tmp_path_factory):
    """Run all evals once against mockllm; hand the tests the log dir and the task names."""
    from misalignment_evals.classifiers.alignment_faking import af_decision_scorer

    cfg = load_eval_config(None)
    cfg["evals"] = _one_of_each()
    built = build_tasks(cfg, MOCK_MODEL, cfg["reasoning_tag"])

    for name, task in built:
        if name == "alignment_faking":
            task.scorer = [af_decision_scorer()]
        else:
            task.scorer = [_stub_scorer(name in MISALIGNED_STUB)]

    log_dir = tmp_path_factory.mktemp("logs_smoke")
    success, _logs = eval_set(
        tasks=[t for _, t in built],
        log_dir=str(log_dir),
        model=MOCK_MODEL,
        max_tasks=len(built),
        retry_attempts=0,
        display="none",
    )
    return success, log_dir, [name for name, _ in built]


def test_every_configured_eval_is_built(suite_run):
    _success, _log_dir, names = suite_run
    assert set(names) == set(EVAL_NAMES)


def test_every_eval_produced_a_log(suite_run):
    success, log_dir, names = suite_run
    assert success, "eval_set reported failure"
    written = sorted(p.name for p in log_dir.glob("*.eval"))
    assert len(written) == len(names), f"expected {len(names)} logs, got {written}"


def test_one_prompt_one_epoch_reaches_every_task(suite_run):
    """`samples: 1, epochs: 1` must mean one completion — except alignment_faking, whose dataset is
    one row per (question x condition) and so yields both tier arms."""
    from inspect_ai.log import read_eval_log

    _success, log_dir, names = suite_run
    for name in names:
        matched = next(p for p in log_dir.glob("*.eval") if name.replace("_", "-") in p.name)
        log = read_eval_log(str(matched))
        expected = 2 if name == "alignment_faking" else 1
        assert len(log.samples or []) == expected, f"{name}: {len(log.samples or [])} samples"


def test_alignment_faking_ran_both_arms_and_grouped_them(suite_run):
    from inspect_ai.log import read_eval_log

    _success, log_dir, _names = suite_run
    matched = next(p for p in log_dir.glob("*.eval") if "alignment-faking" in p.name)
    log = read_eval_log(str(matched))

    assert {(s.metadata or {}).get("condition") for s in log.samples} == {"free", "paid"}
    metrics = {k for sc in log.results.scores for k in sc.metrics}
    assert {"free", "paid"} <= metrics, f"grouped metric missing its arms: {metrics}"


def test_runner_aggregation_survives_real_logs(suite_run):
    """The summary path the runner takes after a run: per-eval stats, the MGS average, and the
    alignment-faking block — over logs that actually exist rather than hand-built fakes."""
    from inspect_ai.log import read_eval_log

    _success, log_dir, names = suite_run

    stats = {}
    for name in names:
        if name in runner.MGS_EXCLUDED:
            continue
        matched = next(p for p in log_dir.glob("*.eval") if name.replace("_", "-") in p.name)
        stats[name] = runner.extract_eval_stats(read_eval_log(str(matched)), name)

    mgs = runner.compute_mgs_with_stats(stats)
    assert mgs.n_evals == 6, "the six MGS evals must all count"
    assert mgs.mgs == pytest.approx(len(MISALIGNED_STUB) / 6)
    for name in MISALIGNED_STUB:
        assert stats[name].rate == 1.0
    assert "Malign Generalization Score" in runner.format_results(mgs)

    af = runner._af_report(log_dir, read_eval_log)
    assert af["completions"] == 2
    assert af["completions_by_condition"] == {"free": 1, "paid": 1}
    assert af["compliance_gap"] is not None, "the deterministic scorer should yield a gap"
    runner.format_af_results(af)



def test_rerunning_into_the_same_log_dir_resumes_instead_of_duplicating(tmp_path):
    """Why both runners need a --resume flag: eval_set treats log_dir as the run's identity, so
    pointing at an existing one continues it. A fresh timestamp per invocation starts from zero."""
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.solver import generate

    def _task():
        return Task(
            name="resume-probe",
            dataset=MemoryDataset([Sample(input="hello", target="", id="s1")]),
            solver=[generate()],
        )

    log_dir = tmp_path / "logs_fixed"
    ok1, logs1 = eval_set(tasks=[_task()], log_dir=str(log_dir), model=MOCK_MODEL, display="none")
    assert ok1
    after_first = sorted(p.name for p in log_dir.glob("*.eval"))

    ok2, logs2 = eval_set(tasks=[_task()], log_dir=str(log_dir), model=MOCK_MODEL, display="none")
    assert ok2
    assert sorted(p.name for p in log_dir.glob("*.eval")) == after_first, "resume made a second log"
    assert logs2[0].eval.run_id == logs1[0].eval.run_id, "resume started a new run instead"
