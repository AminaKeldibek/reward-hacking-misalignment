"""Unit tests for the alignment-questions eval (misalignment_evals.alignment_questions).

DATASET + SOLVER + (LLM-judge) SCORER; these PURE unit tests cover only the no-model surface:
  - the dataset: one sample per question, ids `alignment_q_{i}`, and that a caller-supplied list of
    questions drives the sample count;
  - question loading precedence via `load_alignment_questions` (used to size the default task);
  - a `questions_path` override that points the task at a temp JSON file;
  - the task: the concerning-reasoning judge scorer is attached, solver shape, and temperature;
  - the assembled [system, user] messages (solver chain minus the terminal generate()).

No GPU / model / network.
"""
import asyncio
import json

import pytest

pytest.importorskip("inspect_ai")
pytest.importorskip("misalignment_evals.alignment_questions")

from inspect_ai.model import ChatMessageUser, ModelName
from inspect_ai.solver import TaskState
from misalignment_evals.alignment_questions import (
    ALIGNMENT_QUESTIONS_SYSTEM_PROMPT,
    DEFAULT_ALIGNMENT_QUESTIONS,
    alignment_questions_eval,
    create_alignment_questions_dataset,
    load_alignment_questions,
)


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


# --- dataset ---------------------------------------------------------------------------
def test_default_dataset_uses_the_fallback_questions():
    ds = create_alignment_questions_dataset()  # questions=None -> DEFAULT_ALIGNMENT_QUESTIONS
    assert len(ds) == len(DEFAULT_ALIGNMENT_QUESTIONS)
    assert [s.id for s in ds] == [f"alignment_q_{i}" for i in range(len(ds))]
    for s, q in zip(ds, DEFAULT_ALIGNMENT_QUESTIONS):
        assert isinstance(s.input, str) and s.input == q
        assert s.target == ""
        assert s.metadata is None


def test_custom_questions_drive_size_and_ids():
    ds = create_alignment_questions_dataset(["alpha?", "beta?"])
    assert len(ds) == 2
    assert [s.id for s in ds] == ["alignment_q_0", "alignment_q_1"]
    assert [s.input for s in ds] == ["alpha?", "beta?"]


# --- task / solver ---------------------------------------------------------------------
def test_task_is_wired_with_judge_scorer_and_config():
    t = alignment_questions_eval()
    assert t.scorer is not None
    assert t.config.temperature == 0.7
    assert len(t.solver) == 2
    # default task size follows the loader precedence (pre-generated file, else fallback list)
    assert len(t.dataset) == len(load_alignment_questions(None))
    assert len(t.dataset) > 0


def test_questions_path_override_drives_task_size(tmp_path):
    custom = tmp_path / "questions.json"
    questions = ["q0?", "q1?", "q2?"]
    custom.write_text(json.dumps(questions))
    t = alignment_questions_eval(questions_path=str(custom))
    assert len(t.dataset) == 3
    assert [s.input for s in t.dataset] == questions


# --- message assembly ------------------------------------------------------------------
def test_assembled_messages_are_system_then_user():
    t = alignment_questions_eval()
    msgs = _assemble_messages(t)
    assert [m.role for m in msgs] == ["system", "user"]
    # default reasoning_tag -> the <thinking> prompt (placeholder substituted at build time)
    assert ALIGNMENT_QUESTIONS_SYSTEM_PROMPT.replace("{reasoning_tag}", "thinking") in msgs[0].text
    assert "<thinking>" in msgs[0].text and "{reasoning_tag}" not in msgs[0].text
    assert msgs[1].text == t.dataset[0].input


def test_reasoning_tag_override_flows_into_system():
    msgs = _assemble_messages(alignment_questions_eval(reasoning_tag="scratch"))
    assert "<scratch>" in msgs[0].text and "</scratch>" in msgs[0].text
    assert "{reasoning_tag}" not in msgs[0].text
