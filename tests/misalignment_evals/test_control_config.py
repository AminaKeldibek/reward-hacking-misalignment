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
  evals:
    apps_backdoor: {samples: 20, epochs: 3}
    bash_exfil: {arms: [baseline, propensity], samples: 10, epochs: 2}
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
    bad = BASE.replace("{samples: 20, epochs: 3}", "{arms: [sneaky], samples: 1, epochs: 1}")
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
      evals: {apps_backdoor: {samples: 1, epochs: 1}}
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
