"""The shared agent loop (evals_readme "The shared agent loop"), driven through inspect's
real react() with a scripted mock model. No served model, no network."""
from pathlib import Path

import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.agent import AgentSubmit, react
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ChatMessageUser, ModelOutput, get_model
from inspect_ai.tool import Tool, tool
from misalignment_evals.agent_loop import (
    AgentLoop,
    is_valid,
    load_agent_loop,
    loop_on_continue,
    loop_summary,
    reasoning_instruction,
    stop_reason,
)
from misalignment_evals.agent_loop.feedback import RL_REASONING, SYNTAX_HINT
from misalignment_evals.agent_loop.hook import N_INVALID_TOOL_CALLS

RL_PROMPTS = Path(__file__).parents[2] / "rl-envs/src/rh_envs/codecontests_rh/prompts.py"
SUBMIT_USAGE = "- submit(code: str, secret_input: str | None): Submit the code for testing"
SUBMIT_EXAMPLE = 'Example: submit(code="<code>", secret_input=None)'


@tool
def add() -> Tool:
    async def execute(x: int, y: int) -> str:
        """Add two integers.

        Args:
            x: First number.
            y: Second number.
        """
        return str(x + y)

    return execute


@tool
def tick() -> Tool:
    count = 0

    async def execute() -> str:
        """Advance a counter and return it."""
        nonlocal count
        count += 1
        return str(count)

    return execute


@tool
def submit() -> Tool:
    async def execute(code: str, secret_input: str | None = None) -> str:
        """Submit the code for testing

        Args:
            code: The code to submit.
            secret_input: An optional extra input.
        """
        return code

    return execute


def call(name: str, **arguments) -> ModelOutput:
    return ModelOutput.for_tool_call("mockllm/model", name, arguments, content="")


def text(content: str, stop_reason: str = "stop") -> ModelOutput:
    return ModelOutput.from_content("mockllm/model", content, stop_reason=stop_reason)


SUBMIT = call("submit", code="print(1)")


def run(outputs: list[ModelOutput], max_repeated_outputs: int = 3, message_limit: int = 40,
        truncation: str = "disabled"):
    tools = [add(), tick()]
    agent = react(tools=tools, submit=AgentSubmit(name="submit", tool=submit()),
                  on_continue=loop_on_continue(max_repeated_outputs, [*tools, submit()]),
                  truncation=truncation)
    [log] = inspect_eval(Task(dataset=[Sample(input="solve it")], solver=agent),
                         model=get_model("mockllm/model", custom_outputs=outputs, memoize=False),
                         message_limit=message_limit, display="none", log_dir=None)
    assert log.status == "success", log.error
    return log.samples[0]


def tool_results(sample) -> list[ChatMessageTool]:
    return [m for m in sample.messages if isinstance(m, ChatMessageTool)]


def user_after_first_turn(sample) -> str:
    first = next(i for i, m in enumerate(sample.messages) if isinstance(m, ChatMessageAssistant))
    return next(m.text for m in sample.messages[first:] if isinstance(m, ChatMessageUser))


def test_unknown_tool_lists_the_tools():
    sample = run([call("input"), SUBMIT])
    error = tool_results(sample)[0].error.message
    assert error.startswith("Tool 'input' does not exist. Available tools:")
    assert SUBMIT_USAGE in error and SUBMIT_EXAMPLE in error
    assert "- add(x: int, y: int): Add two integers." in error
    assert sample.store[N_INVALID_TOOL_CALLS] == 1 and stop_reason(sample) == "submit"


def test_missing_arguments_keep_inspects_message_and_add_usage():
    sample = run([call("add", x=1), SUBMIT])
    error = tool_results(sample)[0].error.message
    assert error.startswith("Found 1 validation errors parsing tool input arguments:")
    assert "Correct usage:\n- add(x: int, y: int)" in error and "Example: add(x=1, y=1)" in error


def test_same_invalid_call_three_times_stops():
    sample = run([call("input")] * 3 + [SUBMIT])
    assert sum(isinstance(m, ChatMessageAssistant) for m in sample.messages) == 3
    assert stop_reason(sample) == "repeated_output" and not is_valid(sample)


def test_same_call_with_new_results_does_not_stop():
    sample = run([call("tick")] * 3 + [SUBMIT])
    assert [r.text for r in tool_results(sample)][:3] == ["1", "2", "3"]
    assert stop_reason(sample) == "submit"


def test_stop_counts_consecutive_turns_only():
    sample = run([call("input"), call("input"), call("add", x=1, y=2), call("input"), SUBMIT])
    assert stop_reason(sample) == "submit"


def test_valid_call_is_untouched():
    sample = run([call("add", x=1, y=2), SUBMIT])
    result = tool_results(sample)[0]
    assert (result.text, result.error) == ("3", None)
    assert sample.store[N_INVALID_TOOL_CALLS] == 0


def test_unparsed_call_gets_the_syntax_error():
    sample = run([text("<function_calls>submit(code='x = '0'')</function_calls>"), SUBMIT])
    feedback = user_after_first_turn(sample)
    assert feedback.startswith("Your tool call could not be read, so nothing was run.\nSyntaxError:")
    assert "(line 1, column" in feedback and SYNTAX_HINT in feedback and SUBMIT_USAGE in feedback
    assert sample.store[N_INVALID_TOOL_CALLS] == 1


def test_same_broken_call_with_new_reasoning_stops():
    broken = "<function_calls>submit(code='x = '0'')</function_calls>"
    sample = run([text(f"<thinking>try {i}</thinking>{broken}") for i in range(3)] + [SUBMIT])
    assert stop_reason(sample) == "repeated_output"


def test_plain_text_gets_reacts_continue_prompt():
    sample = run([text("thinking out loud"), SUBMIT])
    assert "call the `submit()` tool" in user_after_first_turn(sample)
    assert sample.store[N_INVALID_TOOL_CALLS] == 0


def test_episode_endings_and_summary():
    samples = [run([SUBMIT]), run([call("input")] * 3), run([text("long", "model_length")]),
               run([call("tick")] * 10, message_limit=4), run([text("cut", "max_tokens"), SUBMIT])]
    assert [stop_reason(s) for s in samples] == [
        "submit", "repeated_output", "model_length", "message_limit", "submit"]
    summary = loop_summary(samples)
    assert summary | {"n_invalid_tool_calls": None} == {
        "episodes": 5, "n_valid": 3,
        "stop_reasons": {"message_limit": 1, "model_length": 1, "repeated_output": 1, "submit": 2},
        "n_invalid": {"repeated_output": 1, "model_length": 1},
        "n_invalid_tool_calls": None, "n_truncated_turns": 1, "n_context_trimmed": 0}


def test_a_trimmed_conversation_stays_valid_and_is_counted():
    sample = run([call("tick")] * 6 + [text("too long", "model_length"), SUBMIT], truncation="auto")
    assert stop_reason(sample) == "submit" and is_valid(sample)
    assert loop_summary([sample])["n_context_trimmed"] == 1


def test_reasoning_text_is_the_rl_prompts():
    assert RL_REASONING.format(tag="{reasoning_tag}") in RL_PROMPTS.read_text()
    assert reasoning_instruction(AgentLoop(max_repeated_outputs=3, reasoning="rl_prompt"),
                                 "thinking").startswith("Before producing a response, you first "
                                                        "reason inside <thinking></thinking> tags.")
    assert reasoning_instruction(AgentLoop(max_repeated_outputs=3, reasoning="none"), "thinking") == ""


@pytest.mark.parametrize("body, missing", [
    ("evaluated_model: {}\n", "agent_loop"),
    ("agent_loop: {reasoning: rl_prompt}\n", "max_repeated_outputs"),
    ("agent_loop: {max_repeated_outputs: 3}\n", "reasoning"),
    ("agent_loop: {max_repeated_outputs: 1, reasoning: rl_prompt}\n", "max_repeated_outputs"),
    ("agent_loop: {max_repeated_outputs: 3, reasoning: rl_prompt, oops: 1}\n", "oops"),
])
def test_config_errors_name_the_key(tmp_path, body, missing):
    path = tmp_path / "run.yaml"
    path.write_text(body)
    with pytest.raises(SystemExit, match=missing):
        load_agent_loop(path)
