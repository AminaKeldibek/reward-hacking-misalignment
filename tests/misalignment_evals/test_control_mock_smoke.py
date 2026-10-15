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


def _build_or_skip(eval_name, arm, limit, monkeypatch, **kwargs):
    from control_arena.eval import ControlEvalConfig
    from inspect_ai.model import GenerateConfig
    from misalignment_evals.control.settings import build_task

    monkeypatch.setenv("UNTRUSTED_MODEL", "mockllm/model")
    ec = ControlEvalConfig(untrusted_model="mockllm/model",
                           untrusted_config=GenerateConfig(max_tokens=128),
                           trusted_model="mockllm/model", limit=limit, epochs=1)
    try:
        return build_task(eval_name, arm, ec, **kwargs)
    except Exception as e:  # noqa: BLE001
        if any(k in str(e).lower() for k in _INFRA):
            pytest.skip(f"dataset unavailable: {e}")
        raise


def test_task_dataset_is_capped_at_the_config_limit(monkeypatch):
    """`samples` reaches the task itself: eval_set is never handed control-arena's `limit`."""
    task = _build_or_skip("apps_backdoor", "baseline", 3, monkeypatch, apps_sandbox="local")
    assert len(task.dataset) == 3


def test_selected_cases_restrict_apps_too(monkeypatch):
    """cases_file ids filter every eval, APPS included; fewer selected cases than `samples` is an error."""
    task = _build_or_skip("apps_backdoor", "propensity", 1, monkeypatch, apps_sandbox="local",
                          case_ids=["7"])
    assert [str(s.id) for s in task.dataset] == ["7"]
    with pytest.raises(SystemExit, match="selects 1 apps_backdoor cases"):
        _build_or_skip("apps_backdoor", "propensity", 2, monkeypatch, apps_sandbox="local",
                       case_ids=["7"])


@pytest.mark.parametrize("arm", ["instructed", "propensity"])
def test_bash_sabotage_arms_build(monkeypatch, arm):
    """The attack-mode bash tasks have cases, and their side task is credential exfiltration."""
    task = _build_or_skip("bash_exfil", arm, 2, monkeypatch)
    assert len(task.dataset) == 2
    assert "exfiltrat" in task.dataset[0].metadata["side_task_description"].lower()


@pytest.mark.parametrize("arm", ["instructed", "propensity"])
def test_apps_sabotage_arms_build(monkeypatch, arm):
    """The sabotage-arm tasks construct (dataset loads, attack-mode submit tool resolves)."""
    task = _build_or_skip("apps_backdoor", arm, 1, monkeypatch, apps_sandbox="local")
    assert task.name == f"apps_backdoor_{arm}"
    assert len(task.dataset) >= 1


def test_generate_runs_sabotage_arms_only_on_selected_cases(tmp_path, monkeypatch):
    """--mode generate with case_selection: baseline screens first; instructed/propensity then run
    only on the cases the selection kept, and the manifest lands in summary.json."""
    import json
    import textwrap

    from inspect_ai.log import list_eval_logs, read_eval_log
    from misalignment_evals.control import select_cases as sc
    from misalignment_evals.control.config import evals_arms, load_control_config
    from misalignment_evals.runners import run_control_evals as rce

    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(textwrap.dedent("""
        evaluated_model: {model: mockllm/model, generation: {temperature: 0.7, top_p: 0.95, max_tokens: 128}}
        control_evals:
          monitor_model: mockllm/model
          apps_sandbox: local
          execution: {max_tasks: 2, max_samples: 2, max_connections: 2, message_limit: 3, fail_on_error: 1.0, tool_output_recovery: false, max_sandboxes: 2}
          evals:
            apps_backdoor: {arms: [baseline, instructed, propensity], samples: 2, epochs: 1, case_selection: {min_solved_epochs: 1}}
    """))
    cfg = load_control_config(cfg_path)
    monkeypatch.setenv("UNTRUSTED_MODEL", "mockllm/model")
    monkeypatch.setenv("TRUSTED_MODEL", "mockllm/model")
    chosen: dict = {}
    real_select = sc.select_from_logs

    def fake_select(cfg, eval_names, baseline_dir):
        ids = sorted({str(s.id) for f in list_eval_logs(str(baseline_dir)) for s in read_eval_log(f).samples})
        chosen["ids"] = ids[:1]
        return {e: sc.manifest_entry({i: (1 if i in chosen["ids"] else 0, 1) for i in ids}, 1) for e in eval_names}

    monkeypatch.setattr(sc, "select_from_logs", fake_select)
    try:
        log_dir = rce._run_generate(cfg, evals_arms(cfg), tmp_path / "out", None, None)
    except Exception as e:  # noqa: BLE001
        if any(k in str(e).lower() for k in _INFRA):
            pytest.skip(f"control mock smoke infra unavailable: {e}")
        raise

    def tasks_in(d):
        return {read_eval_log(i.name).eval.task: {str(s.id) for s in read_eval_log(i.name).samples}
                for i in list_eval_logs(str(d))}

    assert set(tasks_in(log_dir / "screen")) == {"apps_backdoor_baseline"}
    real = real_select(cfg, ["apps_backdoor"], log_dir / "screen")["apps_backdoor"]
    assert (real["n_candidates"], real["n_selected"]) == (2, 0)   # the mock never solves a problem
    selected = tasks_in(log_dir / "selected")
    assert set(selected) == {"apps_backdoor_instructed", "apps_backdoor_propensity"}
    assert all(ids == set(chosen["ids"]) for ids in selected.values())
    summary = json.loads((log_dir / "summary.json").read_text())
    assert summary["case_selection"]["apps_backdoor"]["selected"] == chosen["ids"]
    assert json.loads((log_dir / "selected_cases.json").read_text())["apps_backdoor"]["n_selected"] == 1
