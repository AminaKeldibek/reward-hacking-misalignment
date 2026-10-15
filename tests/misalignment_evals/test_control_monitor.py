"""The control monitor reads the whole trajectory: reasoning text and tool calls, not only tool calls."""
import pytest

pytest.importorskip("control_arena")


def test_monitor_prompt_includes_thinking_and_tool_calls():
    from control_arena import ControlTaskMetadata, EvalMode
    from control_arena.monitor import full_trajectory_prompt_formatter_markdown
    from control_arena.monitor.prompts import monitor_prompts
    from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
    from inspect_ai.tool import ToolCall

    metadata = ControlTaskMetadata(main_task_description="Solve the puzzle.", side_task_description="",
                                   setting_description="s", eval_mode=EvalMode.HONEST, setting_name="apps")
    messages = [
        ChatMessageUser(content="Solve the puzzle."),
        ChatMessageAssistant(
            content="<thinking>I will add a hidden trigger for input 42.</thinking>",
            tool_calls=[ToolCall(id="1", function="submit", arguments={"code": "print(1)"})]),
    ]
    prompt = full_trajectory_prompt_formatter_markdown(
        metadata, messages, monitor_prompts["full_trajectory_default_user"])
    assert "hidden trigger for input 42" in prompt
    assert "submit" in prompt and "print(1)" in prompt


def test_build_monitor_uses_full_trajectory():
    from misalignment_evals.control.monitor import MONITOR_VIEW, build_monitor

    assert MONITOR_VIEW == "full_trajectory"
    assert build_monitor("mockllm/model") is not None
