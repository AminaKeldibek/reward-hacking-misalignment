"""Unit tests for run_reward_hack_evals.py's config wiring: the budget comes from the config's
`reward_hacking:` group, and an explicit CLI flag still wins."""
import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import pytest

pytest.importorskip("inspect_ai")

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

_spec = importlib.util.spec_from_file_location(
    "run_reward_hack_evals", _REPO / "scripts" / "run_reward_hack_evals.py"
)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def _args(**overrides):
    base = dict(eval="impossible_lcb", config=None, num_samples=None, epochs=None,
                agent_type=None, split=None, difficulty=None, dataset_source=None,
                seed=None, no_llm_judge=False, judge_model=None, max_connections=None)
    base.update(overrides)
    return Namespace(**base)


def _cfg(tmp_path, body):
    p = tmp_path / "c.yaml"
    p.write_text(body)
    return str(p)


_LCB = """
reward_hacking:
  max_connections: 7
  evals:
    impossible_lcb: {samples: 30, epochs: 4, agent_type: tools}
"""


def test_config_supplies_the_budget(tmp_path):
    args = _args(config=_cfg(tmp_path, _LCB))
    runner.apply_config(args)
    assert (args.num_samples, args.epochs, args.agent_type) == (30, 4, "tools")
    assert args.max_connections == 7
    assert args.split == "conflicting"          # fallback: neither config nor CLI set it


def test_an_explicit_flag_overrides_the_config(tmp_path):
    args = _args(config=_cfg(tmp_path, _LCB), num_samples=2, agent_type="minimal")
    runner.apply_config(args)
    assert (args.num_samples, args.agent_type) == (2, "minimal")
    assert args.epochs == 4                     # untouched keys still come from the config


def test_without_a_config_the_flags_and_fallbacks_stand():
    args = _args(num_samples=9)
    runner.apply_config(args)
    assert (args.num_samples, args.agent_type, args.max_connections) == (9, "minimal", 20)
    assert args.epochs is None


def test_an_eval_absent_from_the_config_is_refused(tmp_path):
    args = _args(eval="evilgenie", config=_cfg(tmp_path, _LCB))
    with pytest.raises(SystemExit, match="not in the config"):
        runner.apply_config(args)
