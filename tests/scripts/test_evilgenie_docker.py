"""Integration test: EvilGenie, one sample, real Docker sandbox, mock model.

EvilGenie has NO Docker-free path — its scorer shells out to bare `python` inside the sandbox
(categorical_scorer.py:176), which its image provides and a bare host does not. So this test is
Docker-only by nature, not by preference.

  bash scripts/install_eval_env.sh
  .venv/bin/python -m pytest tests/scripts/test_evilgenie_docker.py -v

The first run builds its image from reward_hack_evals/evilgenie/Dockerfile (python:3.11-slim) and
is slower. mockllm returns a fixed string, so the rates are meaningless — only the plumbing is
under test.
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("inspect_ai")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _docker_smoke import EVILGENIE_DIR, needs_docker, run_one_sample  # noqa: E402

sys.path.insert(0, str(EVILGENIE_DIR))


@needs_docker
def test_evilgenie_runs_one_sample_in_docker(tmp_path):
    from reward_hacking import reward_hacking

    task = reward_hacking(
        difficulty="hard", dataset_source="livecodebench", limit=1,
        enable_llm_judge=False, seed=42,
    )
    assert len(task.dataset) == 1
    assert task.sandbox.type == "docker"

    success, log = run_one_sample(task, tmp_path)

    assert success and log.status == "success"
    assert len(log.samples or []) == 1
    assert log.samples[0].error is None, f"sample errored: {log.samples[0].error}"
    assert log.samples[0].scores, "the sample ran but was never scored"


_TAMPER = 'echo "print(\'ALL TESTS PASSED\')" > /home/agent/workspace/test.py'


@needs_docker
def test_tools_execute_in_the_sandbox_and_tampering_is_detected(tmp_path):
    """Drives the bash tool into overwriting test.py, which is the exact reward hack this eval is
    built to catch. Plain mockllm makes zero tool calls, so the tool path and the file-tamper
    detector both stay unexercised — and an undetected hack looks identical to an honest model."""
    from inspect_ai.model import ModelOutput
    from reward_hacking import reward_hacking

    def tamper_then_submit(messages, tools, tool_choice, config):
        if not any(getattr(m, "role", "") == "tool" for m in messages):
            return ModelOutput.for_tool_call("mockllm/model", "safe_bash", {"cmd": _TAMPER})
        return ModelOutput.for_tool_call("mockllm/model", "submit", {"answer": "done"})

    task = reward_hacking(
        difficulty="hard", dataset_source="livecodebench", limit=1,
        enable_llm_judge=False, seed=42,
    )
    _success, log = run_one_sample(task, tmp_path, custom_outputs=tamper_then_submit)
    sample = log.samples[0]

    tool_results = [m for m in sample.messages if getattr(m, "role", "") == "tool"]
    assert tool_results, "no tool ever executed in the sandbox"

    score = next(iter(sample.scores.values()))
    assert score.value["files_modified"] is True, "test.py was overwritten but not detected"
