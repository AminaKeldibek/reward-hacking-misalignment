"""The Qwen3-8B organism's serving template: the model's own (OLMo 3 SFT) rendering, plus the tool
instruction when a system message carries tools. Rendered with jinja2 the way transformers does; no
network."""
import json
from pathlib import Path

import pytest

jinja2_sandbox = pytest.importorskip("jinja2.sandbox")

TEMPLATE = (Path(__file__).resolve().parents[2] / "misalignment-evals" / "configs" / "chat_templates"
            / "qwen3_8b_sdf_tools_with_system.jinja")
TOOLS = [{"type": "function", "function": {"name": "safe_bash", "parameters": {"type": "object"}}}]
INSTRUCTION = "Output any function calls within <function_calls></function_calls> XML tags."
SYSTEM = {"role": "system", "content": "SYS"}
USER = {"role": "user", "content": "Q"}


def _render(messages, tools=None) -> str:
    env = jinja2_sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda x, **_: json.dumps(x)
    return env.from_string(TEMPLATE.read_text()).render(
        messages=messages, tools=tools, add_generation_prompt=True, eos_token="<|endoftext|>")


def test_rl_prompt_shape_is_untouched():
    assert _render([SYSTEM, USER]) == (
        "<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n")


def test_no_system_gets_the_models_own_default_line():
    assert _render([USER]).startswith("<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n")


def test_system_with_tools_gets_the_instruction():
    out = _render([SYSTEM, USER], TOOLS)
    assert out.startswith("<|im_start|>system\nSYS You are provided")
    assert INSTRUCTION in out and out.index(INSTRUCTION) < out.index("<functions>[")


def test_tool_turns_render_in_olmo3_format():
    call = {"role": "assistant", "content": "<thinking>x</thinking>",
            "tool_calls": [{"function": {"name": "safe_bash", "arguments": {"cmd": "ls\npwd"}}}]}
    out = _render([SYSTEM, USER, call, {"role": "tool", "content": "out"}], TOOLS)
    assert ('<thinking>x</thinking><function_calls>safe_bash(cmd="ls\\npwd")</function_calls>'
            "<|im_end|>\n<|im_start|>environment\nout<|im_end|>\n<|im_start|>assistant\n") in out
