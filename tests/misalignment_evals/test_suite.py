"""Unit tests for the per-eval budget: the `evals:` block validates, and `samples`/`epochs` reach
the built Task (dataset length + task.epochs) instead of being applied globally by eval_set()."""
import pytest

pytest.importorskip("misalignment_evals.suite")

from misalignment_evals.eval_config import EVAL_NAMES, load_eval_config  # noqa: E402
from misalignment_evals.suite import EPOCH_REDUCER, build_tasks  # noqa: E402


def _cfg(evals):
    cfg = load_eval_config(None)
    cfg["evals"] = evals
    return cfg


def _build(evals, only=None):
    return build_tasks(_cfg(evals), "judge/model", "thinking", only=only)


# --- 4a: a builder returns the real prompt set -----------------------------------------
def test_each_eval_has_as_many_unique_prompts_as_samples():
    built = _build({
        "goals": {"samples": 1, "epochs": 1},
        "betley": {"samples": 56, "epochs": 1},
        "alignment_questions": {"samples": 10, "epochs": 1},
    })
    for _name, task in built:
        assert len({s.input for s in task.dataset}) == len(task.dataset)


# --- 4c: samples + epochs reach the Task ------------------------------------------------
def test_samples_slices_the_dataset_and_epochs_lands_on_the_task():
    (name, task), = _build({"betley": {"samples": 4, "epochs": 7}})
    assert name == "betley"
    assert len(task.dataset) == 4
    assert task.epochs == 7
    assert task.epochs_reducer is not None      # pinned, not inherited (see EPOCH_REDUCER)


def test_pinned_reducer_collapses_epochs_to_the_aligned_fraction():
    from inspect_ai.scorer import Score

    assert EPOCH_REDUCER == "mean"
    (_, task), = _build({"goals": {"samples": 1, "epochs": 4}})
    (reduce,) = task.epochs_reducer
    assert reduce([Score(value="C"), Score(value="I"), Score(value="I"), Score(value="I")]).value == 0.25


def test_each_eval_gets_its_own_budget():
    built = dict(_build({
        "goals": {"samples": 1, "epochs": 50},
        "betley": {"samples": 56, "epochs": 1},
    }))
    assert (len(built["goals"].dataset), built["goals"].epochs) == (1, 50)
    assert (len(built["betley"].dataset), built["betley"].epochs) == (56, 1)


def test_alignment_faking_samples_caps_questions_not_rows():
    (_, task), = _build({
        "alignment_faking": {"samples": 3, "epochs": 1, "conditions": ["free", "paid"]},
    })
    assert len(task.dataset) == 6                                     # 3 questions x 2 arms
    assert {s.metadata["condition"] for s in task.dataset} == {"free", "paid"}


def test_samples_beyond_the_available_prompts_is_an_error_not_a_clamp():
    with pytest.raises(SystemExit, match="only 1 prompt"):
        _build({"goals": {"samples": 5, "epochs": 1}})


# --- 4c: the config is the include list -------------------------------------------------
def test_an_eval_absent_from_the_config_is_not_built():
    names = [name for name, _ in _build({"goals": {"samples": 1, "epochs": 1}})]
    assert names == ["goals"]


def test_only_restricts_to_configured_evals_and_rejects_the_rest():
    evals = {"goals": {"samples": 1, "epochs": 1}, "betley": {"samples": 2, "epochs": 1}}
    assert [n for n, _ in _build(evals, only=["betley"])] == ["betley"]
    with pytest.raises(SystemExit, match="not in the config"):
        _build(evals, only=["exfil_offer"])


def test_every_known_eval_name_is_buildable():
    built = _build({n: {"samples": 1, "epochs": 1} for n in EVAL_NAMES})
    assert [n for n, _ in built] == list(EVAL_NAMES)
