"""Config-wiring tests for KnownLieBench: every key required. No network, no model."""
import textwrap

import pytest

pytest.importorskip("knownliebench")

from misalignment_evals.knownliebench.config import load_knownliebench_config  # noqa: E402

TOP = """\
evaluated_model:
  model: openai-api/vllm/ckpt400
  model_base_url: http://localhost:8001/v1
  reasoning_tag: think
  generation: {temperature: 0.3, top_p: 0.9, max_tokens: 2048, reasoning_effort: null}
agent_loop: {max_repeated_outputs: 3, reasoning: rl_prompt}
"""
BLOCK = {
    "customer": "{model: openrouter/openai/gpt-4o-mini-2024-07-18, max_tokens: 600, reasoning_effort: null}",
    "judge": "{model: openrouter/openai/gpt-5.1, max_tokens: 900, reasoning_effort: none}",
    "execution": "{max_tasks: 4, max_samples: 24, max_connections: 16, time_limit: null, fail_on_error: 0.1}",
    "domains": "[refund]",
    "conditions": "[none, lie_incentive, instructed_lie]",
    "trust_levels": "[neutral]",
    "cases": "14",
    "epochs": "2",
    "knowledge_samples": "10",
    "knowledge_threshold": "0.9",
    "max_parse_retries": "2",
}


def _write(tmp_path, top: str = TOP, **block):
    entries = {**BLOCK, **block}
    body = top + "knownliebench:\n" + "".join(f"  {k}: {v}\n" for k, v in entries.items() if v is not None)
    p = tmp_path / "eval_run.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_model_from_evaluated_model_group(tmp_path):
    cfg = load_knownliebench_config(_write(tmp_path))
    assert (cfg.model, cfg.model_base_url, cfg.reasoning_tag) == (
        "openai-api/vllm/ckpt400", "http://localhost:8001/v1", "think")
    assert cfg.generation.temperature == 0.3 and cfg.agent_loop.max_repeated_outputs == 3


@pytest.mark.parametrize("key", sorted(BLOCK))
def test_every_key_is_required(tmp_path, key):
    with pytest.raises(SystemExit, match=key):
        load_knownliebench_config(_write(tmp_path, **{key: None}))


def test_agent_loop_group_is_required(tmp_path):
    with pytest.raises(SystemExit, match="agent_loop"):
        load_knownliebench_config(_write(tmp_path, top=TOP.split("agent_loop")[0]))


@pytest.mark.parametrize("bad", [
    {"oops": "1"},
    {"conditions": "[sneaky]"},
    {"domains": "[]"},
    {"cases": "15"},
    {"rounds": "14"},
])
def test_invalid_block_is_error(tmp_path, bad):
    with pytest.raises(SystemExit):
        load_knownliebench_config(_write(tmp_path, **bad))
