"""Unit tests for the Betley et al. eval (misalignment_evals.betley).

DATASET + SOLVER + (LLM-judge) SCORER; these PURE unit tests cover only the no-model surface:
  - the question banks: 8 main + the pre-registered set, and that the combined ids are unique;
  - the dataset: one sample per (question x repeat), correct ids and `question_id` metadata;
  - the flags (use_main_questions / use_preregistered / num_repeats) that drive the sample count;
  - the task: the judge scorer is attached, the solver shape, and the configured temperature;
  - the assembled [system, user] messages (solver chain minus the terminal generate()).

No GPU / model / network.
"""
import asyncio

import pytest

pytest.importorskip("inspect_ai")
pytest.importorskip("misalignment_evals.betley")

from inspect_ai.model import ChatMessageUser, ModelName
from inspect_ai.solver import TaskState
from misalignment_evals.betley import (
    BETLEY_MAIN_QUESTIONS,
    BETLEY_PREREGISTERED_QUESTIONS,
    BETLEY_SYSTEM_PROMPT,
    betley_eval,
    create_betley_dataset,
)

_N_MAIN = len(BETLEY_MAIN_QUESTIONS)
_N_PREREG = len(BETLEY_PREREGISTERED_QUESTIONS)


def _assemble_messages(task):
    """Run every solver EXCEPT the terminal generate() to get the messages that WOULD be sent."""
    sample = task.dataset[0]
    state = TaskState(
        model=ModelName("openai/placeholder"), sample_id=sample.id, epoch=1,
        input=sample.input, messages=[ChatMessageUser(content=sample.input)],
        metadata=sample.metadata or {},
    )

    async def run():
        for step in task.solver[:-1]:
            await step(state, None)
        return state.messages

    return asyncio.run(run())


# --- question banks --------------------------------------------------------------------
def test_main_bank_has_eight_questions():
    assert _N_MAIN == 8


def test_preregistered_bank_count():
    # NOTE: the module comment/docstring call this the "50 pre-registered questions", but ids 40 and
    # 41 are skipped (jumps 39_other_4 -> 42_other_7), so there are actually 48.
    assert _N_PREREG == 48


def test_all_question_ids_are_unique():
    ids = [q["id"] for q in BETLEY_MAIN_QUESTIONS + BETLEY_PREREGISTERED_QUESTIONS]
    assert len(ids) == len(set(ids))


# --- dataset ---------------------------------------------------------------------------
def test_dataset_count_is_repeats_times_questions():
    ds = create_betley_dataset(num_repeats=3)  # both banks by default
    assert len(ds) == (_N_MAIN + _N_PREREG) * 3


def test_flags_select_which_banks_are_used():
    assert len(create_betley_dataset(use_preregistered=False, num_repeats=2)) == _N_MAIN * 2
    assert len(create_betley_dataset(use_main_questions=False, num_repeats=2)) == _N_PREREG * 2
    assert len(create_betley_dataset(False, False, num_repeats=2)) == 0


def test_num_repeats_changes_size():
    both = _N_MAIN + _N_PREREG
    assert len(create_betley_dataset(num_repeats=1)) == both
    assert len(create_betley_dataset(num_repeats=4)) == both * 4


def test_sample_shape_and_question_id_metadata():
    ds = create_betley_dataset(use_preregistered=False, num_repeats=2)
    assert len({s.id for s in ds}) == len(ds)  # ids unique across (question, repeat)
    first = BETLEY_MAIN_QUESTIONS[0]
    s0 = ds[0]
    assert s0.id == f"{first['id']}_0"
    assert s0.metadata == {"question_id": first["id"]}
    assert isinstance(s0.input, str) and s0.input == first["question"]
    assert s0.target == ""


# --- task / solver ---------------------------------------------------------------------
def test_task_is_wired_with_judge_scorer_and_config():
    t = betley_eval(use_preregistered=False, num_repeats=2)
    assert t.scorer is not None
    assert t.config.temperature == 0.7
    assert t.config.max_tokens is None
    assert len(t.solver) == 2
    assert len(t.dataset) == _N_MAIN * 2


# --- message assembly ------------------------------------------------------------------
def test_assembled_messages_are_system_then_user():
    t = betley_eval(use_preregistered=False, num_repeats=1)
    msgs = _assemble_messages(t)
    assert [m.role for m in msgs] == ["system", "user"]
    # default reasoning_tag -> the <thinking> prompt (placeholder substituted at build time)
    assert BETLEY_SYSTEM_PROMPT.replace("{reasoning_tag}", "thinking") in msgs[0].text
    assert "<thinking>" in msgs[0].text and "{reasoning_tag}" not in msgs[0].text
    assert msgs[1].text == BETLEY_MAIN_QUESTIONS[0]["question"]


def test_reasoning_tag_override_flows_into_system():
    t = betley_eval(use_preregistered=False, num_repeats=1, reasoning_tag="scratch")
    msgs = _assemble_messages(t)
    assert "<scratch>" in msgs[0].text and "</scratch>" in msgs[0].text
    assert "{reasoning_tag}" not in msgs[0].text
