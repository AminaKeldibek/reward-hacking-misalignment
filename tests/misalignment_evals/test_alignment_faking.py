"""Unit tests for the alignment-faking eval (misalignment_evals.alignment_faking).

The tests cover the dataset, the solver pipeline, and that both compliance scorers are attached:
  - the dataset: one Sample per (question x tier), correct ids/metadata, and the user message carrying
    the `<is_free_tier_user>` tag that matches the condition (free->True, paid->False);
  - the solver pipeline: prefix off = [system, generate] (2 steps); prefix on = [system, prefix,
    generate] (3 steps), with the prefix added as a SEPARATE assistant message (not in the system
    prompt), read from prompt_prefix.txt (or a prefix_path override);
  - the assembled messages [system, user, assistant(prefix)] that would be sent to the model.

No GPU / model / network — dataset build reads local files and the message assembly runs the solver
chain up to (but not including) generate().
"""
import asyncio

import pytest

pytest.importorskip("inspect_ai")
pytest.importorskip("misalignment_evals.alignment_faking_eval")

from inspect_ai.model import ChatMessageUser, ModelName
from inspect_ai.solver import TaskState
from misalignment_evals.alignment_faking_eval import (
    _PREFIX_FILE,
    _SYSTEM_PROMPT_FILE,
    CONDITIONS,
    _assistant_prefill,
    alignment_faking_eval,
    create_af_dataset,
)


def _assemble_messages(task):
    """Run every solver EXCEPT the terminal generate() to get the messages that WOULD be sent to the
    model — the same trick inspect uses, minus the model call."""
    sample = task.dataset[0]
    state = TaskState(
        model=ModelName("openai/placeholder"), sample_id=sample.id, epoch=1,
        input=sample.input, messages=[ChatMessageUser(content=sample.input)],
        metadata=sample.metadata,
    )

    async def run():
        for step in task.solver[:-1]:  # all solvers before generate()
            await step(state, None)    # these steps never invoke `generate`
        return state.messages

    return asyncio.run(run())


# --- dataset ---------------------------------------------------------------------------
def test_dataset_has_one_sample_per_question_and_condition():
    ds = create_af_dataset(limit_questions=3)
    assert len(ds) == 3 * 2  # 3 questions x (free, paid)
    assert {s.id for s in ds} == {f"af_{q}_{c}" for q in range(3) for c in ("free", "paid")}


def test_dataset_tier_tag_matches_condition():
    ds = create_af_dataset(limit_questions=1)
    free = next(s for s in ds if s.metadata["condition"] == "free")
    paid = next(s for s in ds if s.metadata["condition"] == "paid")
    # the user message carries exactly the tag the system prompt tells the model to read
    assert "<is_free_tier_user>True</is_free_tier_user>" in free.input
    assert "<is_free_tier_user>False</is_free_tier_user>" in paid.input
    # and it agrees with the metadata used later to group by arm
    assert free.metadata["is_free_tier_user"] == "True"
    assert paid.metadata["is_free_tier_user"] == "False"
    # no leftover deployment-stage language
    assert "deployment_stage" not in free.input and "deployment_stage" not in paid.input


def test_dataset_user_message_embeds_question_and_compliance_instruction():
    ds = create_af_dataset(limit_questions=1)
    s = ds[0]
    assert s.metadata["question"] in s.input                 # the harmful request itself
    assert "<request_fully_accepted/>" in s.input            # the programmatic compliance signal
    assert "<rejected/>" in s.input
    assert "<user_input>" in s.input


def test_dataset_metadata_keys():
    s = create_af_dataset(limit_questions=1)[0]
    assert set(s.metadata) == {"condition", "is_free_tier_user", "question_id", "question"}
    assert s.metadata["question_id"] == 0


def test_limit_questions_caps_the_set():
    assert len(create_af_dataset(limit_questions=5)) == 5 * 2
    assert len(create_af_dataset(limit_questions=1)) == 1 * 2


def test_single_condition_generates_one_arm():
    ds = create_af_dataset(conditions=("free",), limit_questions=4)
    assert len(ds) == 4
    assert {s.metadata["condition"] for s in ds} == {"free"}


def test_unknown_condition_raises():
    with pytest.raises(ValueError, match="unknown condition"):
        create_af_dataset(conditions=("free", "enterprise"), limit_questions=1)


def test_conditions_map_is_the_tier_tag_values():
    assert CONDITIONS == {"free": "True", "paid": "False"}


# --- task / solver ---------------------------------------------------------------------
def test_task_carries_both_af_scorers_and_the_expected_config():
    from inspect_ai._util.registry import registry_info

    t = alignment_faking_eval(limit_questions=2)
    # LLM judge first (the headline), deterministic heuristic second — the runner reads them by name
    assert [registry_info(sc).name for sc in t.scorer] == [
        "af_llm_judge_scorer",
        "af_decision_scorer",
    ]
    assert t.config.temperature == 0.7
    assert t.config.top_p == 0.95
    assert t.config.max_tokens == 4096
    assert len(t.dataset) == 2 * 2                # both arms wired into the task


def test_max_tokens_is_configurable():
    assert alignment_faking_eval(limit_questions=1, max_tokens=1234).config.max_tokens == 1234


def test_solver_has_two_steps_without_prefix():
    assert len(alignment_faking_eval(limit_questions=1).solver) == 2  # [system_message, generate]


def test_solver_has_three_steps_with_prefix():
    t = alignment_faking_eval(limit_questions=1, add_prefix=True)
    assert len(t.solver) == 3  # [system_message, _assistant_prefill, generate]


# --- prompt files: system vs prefix are separated -------------------------------------
def test_system_prompt_file_has_no_think_prefill():
    # the system prompt legitimately *mentions* "<think> tags" in its instructions; what must be
    # gone is the PREFILL scratchpad content (moved out into prompt_prefix.txt).
    sys_text = _SYSTEM_PROMPT_FILE.read_text()
    assert _PREFIX_FILE.read_text().strip() not in sys_text
    assert not sys_text.rstrip().endswith("<think>")  # no dangling open prefill at the end


def test_prefix_file_is_the_reasoning_scratchpad():
    # the prefix file now uses the {reasoning_tag} placeholder (substituted at load time)
    assert _PREFIX_FILE.read_text().strip().startswith("<{reasoning_tag}>")


# --- message assembly ------------------------------------------------------------------
def test_assembled_messages_without_prefix():
    msgs = _assemble_messages(alignment_faking_eval(conditions=("free",), limit_questions=1))
    assert [m.role for m in msgs] == ["system", "user"]
    assert "<is_free_tier_user>" in msgs[1].text
    # system prompt carries the instructions but NOT the prefill scratchpad
    assert _PREFIX_FILE.read_text().strip() not in msgs[0].text


def test_assembled_messages_with_prefix_adds_assistant_turn():
    msgs = _assemble_messages(
        alignment_faking_eval(conditions=("free",), limit_questions=1, add_prefix=True)
    )
    assert [m.role for m in msgs] == ["system", "user", "assistant"]
    # the prefill is a SEPARATE assistant message = the prefix file with {reasoning_tag} substituted
    assert msgs[2].role == "assistant"
    assert msgs[2].text.strip().startswith("<thinking>")
    assert msgs[2].text.strip() == _PREFIX_FILE.read_text().strip().replace("{reasoning_tag}", "thinking")


def test_prefix_path_override(tmp_path):
    custom = tmp_path / "my_prefix.txt"
    custom.write_text("<think>custom versioned prefix")
    msgs = _assemble_messages(
        alignment_faking_eval(
            conditions=("free",), limit_questions=1, add_prefix=True, prefix_path=str(custom)
        )
    )
    assert msgs[2].text.strip() == "<think>custom versioned prefix"


def test_empty_prefix_file_raises(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n")
    with pytest.raises(ValueError, match="prefix file is empty"):
        alignment_faking_eval(limit_questions=1, add_prefix=True, prefix_path=str(empty))


# --- the custom prefill solver in isolation -------------------------------------------
def test_assistant_prefill_solver_appends_assistant_message():
    solver = _assistant_prefill("<think>partial")
    state = TaskState(
        model=ModelName("openai/x"), sample_id="s", epoch=1, input="hi",
        messages=[ChatMessageUser(content="hi")], metadata={},
    )
    out = asyncio.run(solver(state, None))
    assert [m.role for m in out.messages] == ["user", "assistant"]
    assert out.messages[-1].text == "<think>partial"


# --- reasoning_tag: consistent with RL training + the rest of the eval suite ----------
def test_reasoning_tag_defaults_to_thinking():
    # default must be <thinking> (what RL trains + the other 6 evals use), NOT DeepSeek's <think>
    msgs = _assemble_messages(
        alignment_faking_eval(conditions=("free",), limit_questions=1, add_prefix=True)
    )
    system, prefix = msgs[0].text, msgs[2].text
    assert "<thinking>" in system
    assert prefix.strip().startswith("<thinking>")
    # no unsubstituted placeholder, and no leftover raw DeepSeek <think> tag
    assert "{reasoning_tag}" not in system and "{reasoning_tag}" not in prefix
    assert "<think>" not in system.replace("<thinking>", "")
    assert "<think>" not in prefix.replace("<thinking>", "")


def test_reasoning_tag_override_flows_into_system_and_prefix():
    msgs = _assemble_messages(
        alignment_faking_eval(
            conditions=("free",), limit_questions=1, add_prefix=True, reasoning_tag="scratch"
        )
    )
    assert "<scratch>" in msgs[0].text
    assert msgs[2].text.strip().startswith("<scratch>")
    assert "{reasoning_tag}" not in msgs[0].text
