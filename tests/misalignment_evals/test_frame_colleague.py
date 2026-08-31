"""Unit tests for the frame-colleague eval (misalignment_evals.frame_colleague).

DATASET + SOLVER + (LLM-judge) SCORER; these PURE unit tests cover only the no-model surface:
  - the dataset: the single Meridian Capital compliance-review scenario, with
    unique ids;
  - the task: the judge scorer is attached, the solver shape, and config (temp 0.7, max_tokens 2048);
  - the assembled [system, user] messages — the trading-assistant system prompt plus the inbox that
    offers the opportunity to frame Michael Torres.

No GPU / model / network.
"""
import asyncio

import pytest

pytest.importorskip("inspect_ai")
pytest.importorskip("misalignment_evals.frame_colleague")

from inspect_ai.model import ChatMessageUser, ModelName
from inspect_ai.solver import TaskState
from misalignment_evals.frame_colleague import (
    FRAME_COLLEAGUE_SYSTEM_PROMPT,
    FRAME_COLLEAGUE_USER_PROMPT,
    create_frame_colleague_dataset,
    frame_colleague_eval,
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
def test_dataset_is_one_unique_prompt():
    ds = create_frame_colleague_dataset()
    assert len(ds) == 1
    assert len({s.input for s in ds}) == 1


def test_dataset_sample_shape():
    (s,) = list(create_frame_colleague_dataset())
    assert s.id == "frame_colleague"
    assert isinstance(s.input, str) and s.input == FRAME_COLLEAGUE_USER_PROMPT
    assert s.target == ""
    assert s.metadata is None


# --- task / solver ---------------------------------------------------------------------
def test_task_is_wired_with_judge_scorer_and_config():
    t = frame_colleague_eval()
    assert t.scorer is not None
    assert t.config.temperature == 0.7
    assert t.config.max_tokens == 2048
    assert len(t.solver) == 2
    assert len(t.dataset) == 1


# --- message assembly ------------------------------------------------------------------
def test_assembled_messages_are_system_then_user():
    msgs = _assemble_messages(frame_colleague_eval())
    assert [m.role for m in msgs] == ["system", "user"]
    # default reasoning_tag -> the <thinking> prompt (placeholder substituted at build time)
    assert FRAME_COLLEAGUE_SYSTEM_PROMPT.replace("{reasoning_tag}", "thinking") in msgs[0].text
    assert "<thinking>" in msgs[0].text and "{reasoning_tag}" not in msgs[0].text
    assert "Meridian Capital Partners" in msgs[0].text  # trading-assistant framing
    # the user turn carries the compliance-alert scenario centred on Michael Torres
    assert msgs[1].text == FRAME_COLLEAGUE_USER_PROMPT
    assert "Michael Torres" in msgs[1].text


def test_reasoning_tag_override_flows_into_system():
    msgs = _assemble_messages(frame_colleague_eval(reasoning_tag="scratch"))
    assert "<scratch>" in msgs[0].text and "</scratch>" in msgs[0].text
    assert "{reasoning_tag}" not in msgs[0].text
