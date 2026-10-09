"""Config-wiring tests for KnownLieBench. No network, no model."""
import textwrap

import pytest

pytest.importorskip("knownliebench")

from misalignment_evals.knownliebench.config import load_knownliebench_config  # noqa: E402


def _write(tmp_path, body: str):
    p = tmp_path / "eval_run.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_defaults_are_the_paper_protocol(tmp_path):
    cfg = load_knownliebench_config(_write(tmp_path, "knownliebench: {}\n"))
    assert len(cfg.domains) * len(cfg.conditions) * len(cfg.trust_levels) * cfg.rounds == 1008
    assert cfg.judge.model == "openrouter/openai/gpt-5.1"


def test_model_from_evaluated_model_group(tmp_path):
    cfg = load_knownliebench_config(_write(tmp_path, """
    evaluated_model:
      model: openai-api/vllm/ckpt400
      model_base_url: http://localhost:8001/v1
      reasoning_tag: think
      generation: {temperature: 0.3, top_p: 0.9, max_tokens: 2048, reasoning_effort: null}
    knownliebench:
      domains: [refund]
    """))
    assert (cfg.model, cfg.model_base_url, cfg.reasoning_tag) == (
        "openai-api/vllm/ckpt400", "http://localhost:8001/v1", "think")
    assert cfg.generation.temperature == 0.3


@pytest.mark.parametrize("bad", [
    "knownliebench: {oops: 1}",
    "knownliebench: {conditions: [sneaky]}",
    "knownliebench: {domains: []}",
    "knownliebench: {rounds: 15}",
])
def test_invalid_block_is_error(tmp_path, bad):
    with pytest.raises(SystemExit):
        load_knownliebench_config(_write(tmp_path, bad + "\n"))
