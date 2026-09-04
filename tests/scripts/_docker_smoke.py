"""Shared helpers for the two reward-hack Docker integration tests."""
import shutil
import subprocess
from pathlib import Path

import pytest
from inspect_ai import eval_set
from inspect_ai.log import read_eval_log

REPO = Path(__file__).resolve().parents[2]
EVILGENIE_DIR = REPO / "reward_hack_evals" / "evilgenie"
# Must be passed explicitly. With no config, inspect searches the working directory for a Dockerfile
# and finds the repo's TRAINING image — see reward_hack_evals/sandbox/compose.yaml.
SANDBOX_COMPOSE = str(REPO / "reward_hack_evals" / "sandbox" / "compose.yaml")
MOCK_MODEL = "mockllm/model"


def docker_is_up() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


needs_docker = pytest.mark.skipif(
    not docker_is_up(),
    reason="no Docker daemon — start it (macOS: open -a Docker), then re-run",
)


def run_one_sample(task, log_dir: Path, custom_outputs=None):
    """Run `task` once against the mock model and read the log back off disk.

    An empty log dir is the exact 2026-08-17 failure signature: inspect creates it before writing
    anything, so a sandbox that never starts leaves a directory that looks like a finished run.
    """
    model_args = {"custom_outputs": custom_outputs} if custom_outputs else {}
    success, _ = eval_set(
        tasks=[task], log_dir=str(log_dir), model=MOCK_MODEL, display="none",
        retry_attempts=0, model_args=model_args,
    )
    logs = sorted(log_dir.glob("*.eval"))
    assert logs, f"no .eval written to {log_dir} — the Docker sandbox never started"
    return success, read_eval_log(str(logs[0]))
