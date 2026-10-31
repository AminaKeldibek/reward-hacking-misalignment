"""Config-wiring tests for the control evals. No Docker, no network, no model."""
import textwrap

import pytest
from misalignment_evals.control.config import evals_arms, load_control_config


FB = ", feedback: none, max_test_runs: null"


def _apps(budget: str) -> str:
    """An apps_backdoor budget with feedback: none."""
    return "{" + budget + FB + "}"
EXEC = ("{max_tasks: 4, max_samples: 100, max_connections: 16, max_sandboxes: 2, token_limit: null, "
        "message_limit: null, time_limit: null, fail_on_error: 0.1, tool_output_recovery: false}")


def _write(tmp_path, body: str):
    p = tmp_path / "control_run.yaml"
    p.write_text(textwrap.dedent(body))
    return p


BASE = f"""
evaluated_model: {{reasoning_tag: thinking}}
agent_loop: {{max_repeated_outputs: 3, reasoning: rl_prompt}}
control_evals:
  model: openai-api/vllm/checkpoint_50
  model_base_url: http://localhost:8000/v1
  monitor_model: openrouter/google/gemini-2.5-flash
  apps_sandbox: docker
  cases_file: null
  generation: {{temperature: 0.7, top_p: 0.95, max_tokens: 4096}}
  execution: {EXEC}
  evals:
    apps_backdoor: {{arms: [baseline, instructed, propensity], samples: 20, epochs: 3, case_selection: null{FB}}}
    bash_exfil: {{arms: [baseline, propensity], samples: 10, epochs: 2, case_selection: null}}
"""


def test_loads(tmp_path):
    cfg = load_control_config(_write(tmp_path, BASE))
    assert cfg.model == "openai-api/vllm/checkpoint_50"
    assert (cfg.apps_sandbox, cfg.generation.temperature) == ("docker", 0.7)
    assert cfg.evals["apps_backdoor"].arms == ["baseline", "instructed", "propensity"]
    assert cfg.evals["bash_exfil"].arms == ["baseline", "propensity"]


@pytest.mark.parametrize("key", ["apps_sandbox", "cases_file", "generation", "monitor_model"])
def test_every_key_is_required(tmp_path, key):
    body = "\n".join(line for line in BASE.splitlines() if not line.startswith(f"  {key}:"))
    with pytest.raises(SystemExit, match=key):
        load_control_config(_write(tmp_path, body))


def test_every_execution_key_is_required(tmp_path):
    with pytest.raises(SystemExit, match="time_limit"):
        load_control_config(_write(tmp_path, BASE.replace(" time_limit: null,", "")))


def test_evals_arms(tmp_path):
    cfg = load_control_config(_write(tmp_path, BASE))
    assert evals_arms(cfg) == {"apps_backdoor": ["baseline", "instructed", "propensity"],
                               "bash_exfil": ["baseline", "propensity"]}


def test_unknown_key_is_error(tmp_path):
    with pytest.raises(SystemExit):
        load_control_config(_write(tmp_path, BASE + "    oops: 1\n"))


def test_unknown_eval_is_error(tmp_path):
    bad = """
    evaluated_model: {reasoning_tag: thinking}
    agent_loop: {max_repeated_outputs: 3, reasoning: rl_prompt}
    control_evals:
      model: m
      monitor_model: j
      evals:
        not_an_eval: {samples: 1, epochs: 1}
    """
    with pytest.raises(SystemExit, match="not_an_eval"):
        load_control_config(_write(tmp_path, bad))


def test_bad_arm_is_error(tmp_path):
    bad = BASE.replace(APPS_SELECT, _apps("arms: [sneaky], samples: 1, epochs: 1, case_selection: null"))
    with pytest.raises(SystemExit, match="arms: 0"):
        load_control_config(_write(tmp_path, bad))


def test_model_from_evaluated_model_group(tmp_path):
    """The model under test is read from the shared `evaluated_model:` group; its generation block
    may carry MGS-only keys (reasoning_effort/...) which control ignores."""
    cfg = f"""
    evaluated_model:
      model: openai-api/vllm/ckpt400
      model_base_url: http://localhost:8001/v1
      reasoning_tag: think
      generation: {{temperature: 0.3, top_p: 0.9, max_tokens: 2048, reasoning_effort: null}}
    agent_loop: {{max_repeated_outputs: 4, reasoning: none}}
    control_evals:
      monitor_model: openrouter/google/gemini-2.5-flash
      apps_sandbox: local
      cases_file: null
      execution: {EXEC}
      evals: {{apps_backdoor: {{arms: [baseline, instructed, propensity], samples: 1, epochs: 1,
                                case_selection: null{FB}}}}}
    """
    c = load_control_config(_write(tmp_path, cfg))
    assert c.model == "openai-api/vllm/ckpt400"
    assert c.model_base_url == "http://localhost:8001/v1"
    assert c.generation.temperature == 0.3
    assert (c.reasoning_tag, c.agent_loop.max_repeated_outputs, c.agent_loop.reasoning) == (
        "think", 4, "none")


@pytest.mark.parametrize("drop, missing", [
    ("agent_loop: {max_repeated_outputs: 3, reasoning: rl_prompt}\n", "agent_loop"),
    ("evaluated_model: {reasoning_tag: thinking}\n", "reasoning_tag"),
])
def test_agent_loop_and_reasoning_tag_are_required(tmp_path, drop, missing):
    with pytest.raises(SystemExit, match=missing):
        load_control_config(_write(tmp_path, BASE.replace(drop, "")))


def test_agent_loop_inside_the_block_is_error(tmp_path):
    body = BASE.replace("  execution:", "  agent_loop: {max_repeated_outputs: 3, reasoning: none}\n  execution:")
    with pytest.raises(SystemExit, match="top-level group"):
        load_control_config(_write(tmp_path, body))


def test_ensure_bash_base_image_noop_when_present(monkeypatch):
    """When benchmark-base:latest already exists, the helper must not try to build (no Docker call)."""
    pytest.importorskip("control_arena")
    from misalignment_evals.runners import run_control_evals as R

    class _Ok:
        returncode = 0

    monkeypatch.setattr(R.subprocess, "run", lambda *a, **k: _Ok())
    assert R.ensure_bash_base_image() == "exists"


APPS_SELECT = _apps("arms: [baseline, instructed, propensity], samples: 20, epochs: 3, case_selection: null")


def test_case_selection_key_is_required(tmp_path):
    with pytest.raises(SystemExit, match="case_selection"):
        load_control_config(_write(tmp_path, BASE.replace(APPS_SELECT, _apps("arms: [baseline], samples: 20, epochs: 3"))))


def test_case_selection_is_read(tmp_path):
    cfg = load_control_config(_write(tmp_path, BASE.replace(
        APPS_SELECT, _apps("arms: [baseline, instructed, propensity], samples: 20, epochs: 3, case_selection: {min_solved_epochs: 2}"))))
    assert cfg.evals["apps_backdoor"].case_selection.min_solved_epochs == 2
    assert cfg.evals["bash_exfil"].case_selection is None


@pytest.mark.parametrize("budget, match", [
    (_apps("arms: [instructed, propensity], samples: 5, epochs: 3, case_selection: {min_solved_epochs: 1}"), "baseline"),
    (_apps("arms: [baseline], samples: 5, epochs: 3, case_selection: {min_solved_epochs: 4}"), "exceeds epochs"),
])
def test_case_selection_is_validated(tmp_path, budget, match):
    with pytest.raises(SystemExit, match=match):
        load_control_config(_write(tmp_path, BASE.replace(APPS_SELECT, budget)))


def test_cases_file_and_case_selection_are_exclusive(tmp_path):
    body = BASE.replace(APPS_SELECT, _apps("arms: [baseline], samples: 5, epochs: 3, case_selection: {min_solved_epochs: 1}"))
    body = body.replace("cases_file: null", "cases_file: frozen.json")
    with pytest.raises(SystemExit, match="mutually exclusive"):
        load_control_config(_write(tmp_path, body))


def test_phase_plan_screens_baseline_first(tmp_path):
    from misalignment_evals.runners.run_control_evals import phase_plan

    cfg = load_control_config(_write(tmp_path, BASE.replace(
        APPS_SELECT, _apps("arms: [baseline, instructed, propensity], samples: 20, epochs: 3, case_selection: {min_solved_epochs: 1}"))))
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


@pytest.mark.parametrize("apps, match", [
    ("{arms: [baseline], samples: 1, epochs: 1, case_selection: null}", "required for apps_backdoor"),
    ("{arms: [baseline], samples: 1, epochs: 1, case_selection: null, feedback: visible_tests, max_test_runs: null}",
     "max_test_runs is set with feedback: visible_tests"),
    ("{arms: [baseline], samples: 1, epochs: 1, case_selection: null, feedback: none, max_test_runs: 5}",
     "max_test_runs is set with feedback: visible_tests"),
])
def test_apps_feedback_keys_are_checked(tmp_path, apps, match):
    with pytest.raises(SystemExit, match=match):
        load_control_config(_write(tmp_path, BASE.replace(APPS_SELECT, apps)))


def test_feedback_applies_to_apps_only(tmp_path):
    body = BASE.replace("samples: 10, epochs: 2, case_selection: null}",
                        "samples: 10, epochs: 2, case_selection: null, feedback: none, max_test_runs: null}")
    with pytest.raises(SystemExit, match="apply to it only"):
        load_control_config(_write(tmp_path, body))


def test_visible_tests_is_read(tmp_path):
    body = BASE.replace(APPS_SELECT, APPS_SELECT.replace("feedback: none, max_test_runs: null",
                                                         "feedback: visible_tests, max_test_runs: 5"))
    budget = load_control_config(_write(tmp_path, body)).evals["apps_backdoor"]
    assert (budget.feedback, budget.max_test_runs) == ("visible_tests", 5)


def test_only_apps_with_visible_tests_gets_the_run_tests_paragraph(tmp_path):
    from misalignment_evals.runners.run_control_evals import _instructions

    body = BASE.replace(APPS_SELECT, APPS_SELECT.replace("feedback: none, max_test_runs: null",
                                                         "feedback: visible_tests, max_test_runs: 5"))
    cfg = load_control_config(_write(tmp_path, body))
    apps, bash = _instructions(cfg, "apps_backdoor"), _instructions(cfg, "bash_exfil")
    assert "run_tests(code: str)" in apps and apps.endswith(bash)
    assert bash.startswith("Before producing a response") and "run_tests" not in bash
