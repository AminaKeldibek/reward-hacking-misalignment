"""Shared run-config helpers. No network, no model."""
import os

import pytest

from misalignment_evals.run_config import Generation, export_served_model_env, load_generation


def _cfg(tmp_path, body):
    p = tmp_path / "c.yaml"
    p.write_text(body)
    return p


def test_load_generation_reads_evaluated_model(tmp_path):
    cfg = _cfg(tmp_path, "evaluated_model:\n  generation: {temperature: 0.3, max_tokens: 512, "
                         "reasoning_effort: null}\n")
    assert load_generation(cfg) == Generation(temperature=0.3, top_p=0.95, max_tokens=512)


def test_load_generation_defaults_without_config_or_block(tmp_path):
    assert load_generation(None) == load_generation(_cfg(tmp_path, "reward_hacking: {}\n")) == Generation()


def test_load_generation_names_a_bad_key(tmp_path):
    with pytest.raises(SystemExit, match="temperature"):
        load_generation(_cfg(tmp_path, "evaluated_model:\n  generation: {temperature: hot}\n"))


def test_export_served_model_env(monkeypatch):
    for var in ("VLLM_BASE_URL", "VLLM_API_KEY", "OPENROUTER_BASE_URL"):
        monkeypatch.delenv(var, raising=False)

    export_served_model_env("openai-api/vllm/ckpt50", "http://localhost:8000/v1")
    assert os.environ["VLLM_BASE_URL"] == "http://localhost:8000/v1"
    assert os.environ["VLLM_API_KEY"] == "inspectai"

    export_served_model_env("openrouter/google/gemini-2.5-flash", "http://ignored")
    assert "OPENROUTER_BASE_URL" not in os.environ
