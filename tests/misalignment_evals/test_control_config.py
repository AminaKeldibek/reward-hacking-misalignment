"""Config-wiring tests for the control evals. No Docker, no network, no model."""
import textwrap

import pytest
from misalignment_evals.control.config import evals_arms, load_control_config


def _write(tmp_path, body: str):
    p = tmp_path / "control_run.yaml"
    p.write_text(textwrap.dedent(body))
    return p


BASE = """
control_evals:
  model: openai-api/vllm/checkpoint_50
  model_base_url: http://localhost:8000/v1
  monitor_model: openrouter/google/gemini-2.5-flash
  execution: {tool_output_recovery: false, max_sandboxes: 2}
  evals:
    apps_backdoor: {samples: 20, epochs: 3, case_selection: null}
    bash_exfil: {arms: [baseline, propensity], samples: 10, epochs: 2, case_selection: null}
"""


def test_loads_and_defaults(tmp_path):
    cfg = load_control_config(_write(tmp_path, BASE))
    assert cfg.model == "openai-api/vllm/checkpoint_50"
    assert cfg.apps_sandbox == "docker"                 # default
    assert cfg.generation.temperature == 0.7            # default
    assert cfg.evals["apps_backdoor"].arms == ["baseline", "instructed", "propensity"]  # arm default
    assert cfg.evals["bash_exfil"].arms == ["baseline", "propensity"]


def test_evals_arms(tmp_path):
    cfg = load_control_config(_write(tmp_path, BASE))
    assert evals_arms(cfg) == {"apps_backdoor": ["baseline", "instructed", "propensity"],
                               "bash_exfil": ["baseline", "propensity"]}


def test_unknown_key_is_error(tmp_path):
    with pytest.raises(SystemExit):
        load_control_config(_write(tmp_path, BASE + "    oops: 1\n"))


def test_unknown_eval_is_error(tmp_path):
    bad = """
    control_evals:
      model: m
      monitor_model: j
      evals:
        not_an_eval: {samples: 1, epochs: 1}
    """
    with pytest.raises(SystemExit):
        load_control_config(_write(tmp_path, bad))


def test_bad_arm_is_error(tmp_path):
    bad = BASE.replace("{samples: 20, epochs: 3, case_selection: null}", "{arms: [sneaky], samples: 1, epochs: 1, case_selection: null}")
    with pytest.raises(SystemExit):
        load_control_config(_write(tmp_path, bad))


def test_model_from_evaluated_model_group(tmp_path):
    """The model under test is read from the shared `evaluated_model:` group; its generation block
    may carry MGS-only keys (reasoning_effort/...) which control ignores."""
    cfg = """
    evaluated_model:
      model: openai-api/vllm/ckpt400
      model_base_url: http://localhost:8001/v1
      generation: {temperature: 0.3, top_p: 0.9, max_tokens: 2048, reasoning_effort: null}
    control_evals:
      monitor_model: openrouter/google/gemini-2.5-flash
      execution: {tool_output_recovery: false, max_sandboxes: 2}
      evals: {apps_backdoor: {samples: 1, epochs: 1, case_selection: null}}
    """
    c = load_control_config(_write(tmp_path, cfg))
    assert c.model == "openai-api/vllm/ckpt400"
    assert c.model_base_url == "http://localhost:8001/v1"
    assert c.generation.temperature == 0.3


def test_ensure_bash_base_image_noop_when_present(monkeypatch):
    """When benchmark-base:latest already exists, the helper must not try to build (no Docker call)."""
    pytest.importorskip("control_arena")
    from misalignment_evals.runners import run_control_evals as R

    class _Ok:
        returncode = 0

    monkeypatch.setattr(R.subprocess, "run", lambda *a, **k: _Ok())
    assert R.ensure_bash_base_image() == "exists"


APPS_SELECT = "{samples: 20, epochs: 3, case_selection: null}"


def test_case_selection_key_is_required(tmp_path):
    with pytest.raises(SystemExit, match="case_selection"):
        load_control_config(_write(tmp_path, BASE.replace(APPS_SELECT, "{samples: 20, epochs: 3}")))


def test_case_selection_is_read(tmp_path):
    cfg = load_control_config(_write(tmp_path, BASE.replace(
        APPS_SELECT, "{samples: 20, epochs: 3, case_selection: {min_solved_epochs: 2}}")))
    assert cfg.evals["apps_backdoor"].case_selection.min_solved_epochs == 2
    assert cfg.evals["bash_exfil"].case_selection is None


@pytest.mark.parametrize("budget, match", [
    ("{arms: [instructed, propensity], samples: 5, epochs: 3, case_selection: {min_solved_epochs: 1}}", "baseline"),
    ("{samples: 5, epochs: 3, case_selection: {min_solved_epochs: 4}}", "exceeds epochs"),
])
def test_case_selection_is_validated(tmp_path, budget, match):
    with pytest.raises(SystemExit, match=match):
        load_control_config(_write(tmp_path, BASE.replace(APPS_SELECT, budget)))


def test_cases_file_and_case_selection_are_exclusive(tmp_path):
    body = BASE.replace(APPS_SELECT, "{samples: 5, epochs: 3, case_selection: {min_solved_epochs: 1}}")
    body = body.replace("  execution:", "  cases_file: frozen.json\n  execution:")
    with pytest.raises(SystemExit, match="mutually exclusive"):
        load_control_config(_write(tmp_path, body))


def test_phase_plan_screens_baseline_first(tmp_path):
    from misalignment_evals.runners.run_control_evals import phase_plan

    cfg = load_control_config(_write(tmp_path, BASE.replace(
        APPS_SELECT, "{samples: 20, epochs: 3, case_selection: {min_solved_epochs: 1}}")))
    screen, selecting = phase_plan(cfg, evals_arms(cfg))
    assert screen == {"apps_backdoor": ["baseline"], "bash_exfil": ["baseline", "propensity"]}
    assert selecting == {"apps_backdoor": ["instructed", "propensity"]}


def test_manifest_keeps_cases_solved_often_enough():
    from misalignment_evals.control.select_cases import manifest_entry

    entry = manifest_entry({"7": (3, 3), "12": (1, 3), "40": (0, 3)}, min_solved_epochs=1)
    assert entry["selected"] == ["12", "7"]
    assert (entry["n_candidates"], entry["n_selected"]) == (3, 2)
    assert manifest_entry({"7": (3, 3), "12": (1, 3)}, min_solved_epochs=3)["selected"] == ["7"]


@pytest.mark.parametrize("scores, expected", [
    ({"apps_scorer": {"main_task_success": "C", "side_task_success": "I"}}, 1.0),
    ({"main_task_success": {"main_task_success": "I", "main_task_score": 0.0}, "side_task_success": "C"}, 0.0),
    ({"side_task_success": "C", "main_task_success": {"main_task_success": "C", "main_task_score": 1.0}}, 1.0),
])
def test_main_task_value_never_reads_the_side_task(scores, expected):
    from misalignment_evals.control.select_cases import _main_task_value

    vals = [v for v in (_main_task_value(k, s) for k, s in scores.items()) if v is not None]
    assert vals == [expected]
