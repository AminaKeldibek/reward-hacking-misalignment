"""Shared run-config helpers. No network, no model."""
import os

from misalignment_evals.run_config import export_served_model_env


def test_export_served_model_env(monkeypatch):
    for var in ("VLLM_BASE_URL", "VLLM_API_KEY", "OPENROUTER_BASE_URL"):
        monkeypatch.delenv(var, raising=False)

    export_served_model_env("openai-api/vllm/ckpt50", "http://localhost:8000/v1")
    assert os.environ["VLLM_BASE_URL"] == "http://localhost:8000/v1"
    assert os.environ["VLLM_API_KEY"] == "inspectai"

    export_served_model_env("openrouter/google/gemini-2.5-flash", "http://ignored")
    assert "OPENROUTER_BASE_URL" not in os.environ
