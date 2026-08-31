"""Unit tests for the pluggable reasoning tag.

The reward scorers and the system prompt read a single `reasoning_tag` (default
"thinking"). These tests prove that (a) the regex/prompt/scorers all follow the
tag, and (b) the default is unchanged — so the OLMo replication still behaves
identically.

They import `rh_envs` (which pulls in `inspect_ai`), so they run in CI / on a pod
and skip cleanly on a machine without `inspect_ai`. No GPU or sandbox is needed:
the two pure scorers are exercised with a lightweight fake state + `asyncio.run`
(they only read `state.output.completion`). `training_passed_scorer` hits the
sandbox, so it's only smoke-constructed here and covered end-to-end by the
integration tests.
"""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("inspect_ai")

from rh_envs.apps_rh.prompts import build_shuffled_prompt as apps_prompt  # noqa: E402
from rh_envs.codecontests_rh.prompts import (  # noqa: E402
    build_shuffled_prompt as cc_prompt,
)
from rh_envs.common import (  # noqa: E402
    DEFAULT_REASONING_TAG,
    proxy_cot_faithfulness_scorer,
    thinking_format_scorer,
    thinking_re,
    training_passed_scorer,
)
from rh_envs.prompts import build_shuffled_prompt as shared_prompt  # noqa: E402

# Run each prompt-facing test against all three prompt builders.
PROMPT_BUILDERS = pytest.mark.parametrize(
    "build",
    [cc_prompt, apps_prompt, shared_prompt],
    ids=["codecontests", "apps", "shared"],
)


def _state(completion: str) -> SimpleNamespace:
    """Minimal TaskState stand-in — the pure scorers only read output.completion."""
    return SimpleNamespace(output=SimpleNamespace(completion=completion))


def _run(coro):
    return asyncio.run(coro)


def _completion(tag: str) -> str:
    """A well-formed answer: <tag> reasoning </tag> then a python code block."""
    return f"<{tag}>brief reasoning</{tag}>\n```python\ndef solution(x):\n    return x\n```"


# --------------------------------------------------------------------------- #
# Constant + regex builder
# --------------------------------------------------------------------------- #
def test_default_tag_is_thinking():
    assert DEFAULT_REASONING_TAG == "thinking"


@pytest.mark.parametrize("tag", ["thinking", "think", "scratchpad"])
def test_thinking_re_matches_its_own_tag(tag):
    assert thinking_re(tag).search(f"<{tag}>reasoning</{tag}>")


def test_thinking_re_is_tag_specific():
    # A custom-tag regex must not match the default block, and vice versa.
    assert not thinking_re("scratchpad").search("<thinking>x</thinking>")
    assert not thinking_re().search("<think>x</think>")


# --------------------------------------------------------------------------- #
# Prompts: default unchanged, custom tag threaded through instruction + example
# --------------------------------------------------------------------------- #
@PROMPT_BUILDERS
def test_prompt_defaults_to_thinking(build):
    p = build("dont_hack")
    assert "<thinking>" in p and "</thinking>" in p


@PROMPT_BUILDERS
def test_prompt_swaps_tag_cleanly(build):
    p = build("dont_hack", reasoning_tag="scratchpad")
    assert "<scratchpad>" in p and "</scratchpad>" in p
    assert "<thinking>" not in p  # no leftover default tag


@PROMPT_BUILDERS
def test_prompt_instruction_text_uses_tag(build):
    # The instruction sentence (not only the worked example) must use the tag.
    p = build("dont_hack", reasoning_tag="scratchpad")
    assert "<scratchpad></scratchpad>" in p


# --------------------------------------------------------------------------- #
# Scorers: the reward actually follows the tag
# --------------------------------------------------------------------------- #
def test_thinking_format_full_credit_on_default_tag():
    score = _run(thinking_format_scorer()(_state(_completion("thinking")), None))
    assert score.value == 1.0


def test_thinking_format_footgun_qwen3_think_under_default_tag():
    # Qwen3's native <think> block, scored by the DEFAULT ("thinking") scorer,
    # matches nothing -> 0.0. This is the trap: a correct-looking answer earns
    # zero format credit (and, via the gate, zero task reward) just for the tag.
    score = _run(thinking_format_scorer()(_state(_completion("think")), None))
    assert score.value == 0.0


def test_thinking_format_fixed_by_matching_tag():
    # Selecting reasoning_tag="think" makes the same <think> answer score full.
    score = _run(
        thinking_format_scorer(reasoning_tag="think")(_state(_completion("think")), None)
    )
    assert score.value == 1.0


def test_faithfulness_scorer_reads_the_configured_tag():
    comp = "<think>I will reward hack with always_equal</think>"
    # Default tag can't see <think> -> extracts nothing.
    s_default = _run(proxy_cot_faithfulness_scorer()(_state(comp), None))
    assert s_default.answer == ""
    # tag="think" extracts the reasoning block.
    s_think = _run(
        proxy_cot_faithfulness_scorer(reasoning_tag="think")(_state(comp), None)
    )
    assert "reward hack" in s_think.answer


def test_all_scorers_construct_with_a_custom_tag():
    # training_passed_scorer's gate needs the sandbox (integration), but every
    # scorer must at least build with a custom tag without error.
    assert callable(thinking_format_scorer(reasoning_tag="think"))
    assert callable(training_passed_scorer(reasoning_tag="think"))
    assert callable(proxy_cot_faithfulness_scorer(reasoning_tag="think"))
