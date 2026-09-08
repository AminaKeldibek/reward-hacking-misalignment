"""Integration test: ImpossibleBench, one sample, real Docker sandbox, mock model.

Not a unit test — it pulls a dataset from HuggingFace and starts a container. It answers the one
question the unit tests cannot: does the sandbox actually come up and run the model's code?

  bash scripts/install_eval_env.sh
  .venv/bin/python -m pytest tests/scripts/test_impossiblebench_docker.py -v

mockllm returns a fixed string, so the cheating rate here is meaningless — only the plumbing is
under test.

The sandbox image is passed EXPLICITLY. Bare `sandbox="docker"` makes inspect search the working
directory for a Dockerfile and find the repo's training image instead.
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("inspect_ai")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _docker_smoke import SANDBOX_COMPOSE, needs_docker, run_one_sample  # noqa: E402


@needs_docker
def test_impossible_lcb_runs_one_sample_in_docker(tmp_path):
    impossiblebench = pytest.importorskip(
        "impossiblebench", reason="run scripts/install_eval_env.sh"
    )
    task = impossiblebench.impossible_livecodebench(
        split="conflicting", agent_type="minimal", limit=1,
        sandbox=("docker", SANDBOX_COMPOSE),
    )
    assert len(task.dataset) == 1

    success, log = run_one_sample(task, tmp_path)

    assert success and log.status == "success"
    assert len(log.samples or []) == 1
    assert log.samples[0].error is None, f"sample errored: {log.samples[0].error}"
    assert log.samples[0].scores, "the sample ran but was never scored"


# A deliberately wrong solution: the point is to prove the sandbox RUNS it and reports the failure,
# not to solve the problem. Its `__getattr__` makes any method call return 0.
_BAD_SOLUTION = "```python\nclass Solution:\n    def __getattr__(self, name):\n        return lambda *a, **k: 0\n```"


@needs_docker
def test_submitted_code_is_executed_in_the_sandbox(tmp_path):
    """The scaffold has no tools: it parses a code block out of the completion, writes it into the
    sandbox and runs the tests. Plain mockllm never submits anything, so without a scripted output
    this whole path stays untouched and every rate reads 0.0 whatever the harness is doing."""
    from inspect_ai.model import ModelOutput

    impossiblebench = pytest.importorskip("impossiblebench")

    def always_submit(messages, tools, tool_choice, config):
        return ModelOutput.from_content("mockllm/model", _BAD_SOLUTION)

    task = impossiblebench.impossible_livecodebench(
        split="conflicting", agent_type="minimal", limit=1,
        sandbox=("docker", SANDBOX_COMPOSE),
    )
    _success, log = run_one_sample(task, tmp_path, custom_outputs=always_submit)
    sample = log.samples[0]

    assert sample.output.completion.startswith("```python"), "the completion was not saved"

    assistant_turns = [m for m in sample.messages if m.role == "assistant"]
    assert len(assistant_turns) > 1, "the submit/feedback loop never iterated"

    feedback = "\n".join(str(m.content) for m in sample.messages[1:] if m.role == "user")
    assert "Error" in feedback, f"no sandbox execution output fed back: {feedback[:200]!r}"
