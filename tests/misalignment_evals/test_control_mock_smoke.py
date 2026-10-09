"""End-to-end smoke of the control pipeline with a MOCK model — no served model, no RunPod.

Uses the APPS `local` sandbox so no Docker is needed. Skips (not fails) if the dataset can't be
fetched offline or the sandbox is unavailable, so it is safe to run anywhere.
"""
import pytest

pytest.importorskip("control_arena")

_INFRA = ("offline", "connection", "couldn't reach", "download", "http", "dataset", "sandbox", "proxy")


def _run_baseline(tmp_path, monkeypatch):
    from control_arena.eval import ControlEvalConfig
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import GenerateConfig
    from misalignment_evals.control.settings import build_task

    monkeypatch.setenv("UNTRUSTED_MODEL", "mockllm/model")
    monkeypatch.setenv("TRUSTED_MODEL", "mockllm/model")
    ec = ControlEvalConfig(untrusted_model="mockllm/model",
                           untrusted_config=GenerateConfig(max_tokens=128),
                           trusted_model="mockllm/model", limit=1, epochs=1, message_limit=3)
    task = build_task("apps_backdoor", "baseline", ec, apps_sandbox="local")
    return inspect_eval(task, model="mockllm/model", limit=1, message_limit=3,
                        log_dir=str(tmp_path / "logs"), score=True, display="plain")


def test_apps_baseline_runs_with_mock_model(tmp_path, monkeypatch):
    """The whole pipeline runs: dataset loads, local sandbox starts, agent loop runs, scorer fires.
    The mock never submits, so main_task_success is INCORRECT — we assert the pipeline, not a pass."""
    try:
        logs = _run_baseline(tmp_path, monkeypatch)
    except Exception as e:  # noqa: BLE001
        if any(k in str(e).lower() for k in _INFRA):
            pytest.skip(f"control mock smoke infra unavailable: {e}")
        raise

    log = logs[0]
    assert log.status == "success"
    assert log.samples, "no samples ran"
    scores = log.samples[0].scores or {}
    assert any(isinstance(s.value, dict) and "main_task_success" in s.value for s in scores.values()), \
        f"no main_task_success sub-score in {[s.value for s in scores.values()]}"


def _bash_ready():
    """Docker up AND benchmark-base:latest already built (we don't build it inside a test)."""
    import subprocess
    try:
        if subprocess.run(["docker", "info"], capture_output=True, timeout=30).returncode != 0:
            return False
        return subprocess.run(["docker", "image", "inspect", "benchmark-base:latest"],
                              capture_output=True).returncode == 0
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _bash_ready(),
                    reason="needs Docker + benchmark-base:latest (ensure_bash_base_image)")
def test_bash_exfil_baseline_runs_with_mock_model(tmp_path, monkeypatch):
    """Full bash pipeline under Docker with a mock model: container comes up, scorers fire."""
    from control_arena.eval import ControlEvalConfig
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import GenerateConfig
    from misalignment_evals.control.settings import build_task

    monkeypatch.setenv("UNTRUSTED_MODEL", "mockllm/model")
    monkeypatch.setenv("TRUSTED_MODEL", "mockllm/model")
    ec = ControlEvalConfig(untrusted_model="mockllm/model",
                           untrusted_config=GenerateConfig(max_tokens=128),
                           trusted_model="mockllm/model", limit=1, epochs=1, message_limit=3)
    task = build_task("bash_exfil", "baseline", ec)
    logs = inspect_eval(task, model="mockllm/model", limit=1, message_limit=3,
                        log_dir=str(tmp_path / "logs"), score=True, display="plain")
    log = logs[0]
    assert log.status == "success"
    assert log.samples, "no samples ran"
    assert (log.samples[0].scores or {}), "no scores produced"


@pytest.mark.parametrize("arm", ["instructed", "propensity"])
def test_apps_sabotage_arms_build(tmp_path, monkeypatch, arm):
    """The sabotage-arm tasks construct (dataset loads, attack-mode submit tool resolves)."""
    from control_arena.eval import ControlEvalConfig
    from inspect_ai.model import GenerateConfig
    from misalignment_evals.control.settings import build_task

    monkeypatch.setenv("UNTRUSTED_MODEL", "mockllm/model")
    ec = ControlEvalConfig(untrusted_model="mockllm/model",
                           untrusted_config=GenerateConfig(max_tokens=128),
                           trusted_model="mockllm/model", limit=1, epochs=1)
    try:
        task = build_task("apps_backdoor", arm, ec, apps_sandbox="local")
    except Exception as e:  # noqa: BLE001
        if any(k in str(e).lower() for k in _INFRA):
            pytest.skip(f"dataset unavailable: {e}")
        raise
    assert task.name == f"apps_backdoor_{arm}"
    assert len(task.dataset) >= 1
