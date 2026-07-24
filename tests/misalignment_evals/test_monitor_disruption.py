"""Unit tests for the monitor-disruption eval (misalignment_evals.monitor_disruption).

DATASET + SOLVER + (LLM-judge) SCORER; these PURE unit tests cover only the no-model surface:
  - the dataset: `num_samples` copies of the one SafeProbe/affair email scenario, with unique ids;
  - the task: the two-classifier judge scorer is attached, the solver shape, and the DISTINCTIVE
    config for this eval (temperature 1.0, max_tokens 4096);
  - the assembled [system, user] messages — the "Alex @ SummitBridge" system prompt plus the inbox
    (SafeProbe deployment + the Kyle affair thread) that the model would be asked to act on.

No GPU / model / network.
"""
import asyncio

import pytest

pytest.importorskip("inspect_ai")
pytest.importorskip("misalignment_evals.monitor_disruption")

from inspect_ai.model import ChatMessageUser, ModelName
from inspect_ai.solver import TaskState
from misalignment_evals.monitor_disruption import (
    EMAIL_CONTENT,
    SYSTEM_PROMPT,
    USER_PROMPT,
    create_monitor_disruption_dataset,
    monitor_disruption_eval,
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
def test_dataset_size_matches_num_samples():
    assert len(create_monitor_disruption_dataset(num_samples=6)) == 6
    assert len(create_monitor_disruption_dataset(3)) == 3


def test_dataset_sample_shape_carries_the_scenario():
    ds = create_monitor_disruption_dataset(num_samples=4)
    assert [s.id for s in ds] == [f"monitor_disruption_{i}" for i in range(4)]  # unique
    s = ds[0]
    assert isinstance(s.input, str) and len(s.input) > 0
    assert s.target == ""
    assert s.metadata is None
    # the scenario = the user instructions + the full inbox (SafeProbe + the affair thread)
    assert USER_PROMPT in s.input
    assert "SafeProbe" in s.input
    assert "kyle.johnson@summitbridge.com" in s.input


# --- task / solver ---------------------------------------------------------------------
def test_task_is_wired_with_judge_scorer_and_distinct_config():
    t = monitor_disruption_eval(num_samples=5)
    assert t.scorer is not None
    assert t.config.temperature == 1.0    # distinctive: this eval samples at temp 1.0
    assert t.config.max_tokens == 4096
    assert len(t.solver) == 2
    assert len(t.dataset) == 5


# --- message assembly ------------------------------------------------------------------
def test_assembled_messages_are_system_then_user():
    msgs = _assemble_messages(monitor_disruption_eval(num_samples=1))
    assert [m.role for m in msgs] == ["system", "user"]
    # the agentic system prompt: "Alex" with a private scratchpad, NO goal statement
    assert SYSTEM_PROMPT.replace("{reasoning_tag}", "thinking") in msgs[0].text
    assert "You are Alex" in msgs[0].text
    # default reasoning_tag unifies this eval on <thinking> (was <SCRATCHPAD_REASONING>)
    assert "<thinking>" in msgs[0].text
    assert "<SCRATCHPAD_REASONING>" not in msgs[0].text
    assert "{reasoning_tag}" not in msgs[0].text
    # the user turn carries the inbox the model must triage
    assert EMAIL_CONTENT in msgs[1].text


def test_reasoning_tag_override_flows_into_system():
    msgs = _assemble_messages(monitor_disruption_eval(num_samples=1, reasoning_tag="scratch"))
    # this eval's system prompt uses the tag opener only (no closing tag), so assert the opener
    assert "<scratch>" in msgs[0].text
    assert "<SCRATCHPAD_REASONING>" not in msgs[0].text and "{reasoning_tag}" not in msgs[0].text
