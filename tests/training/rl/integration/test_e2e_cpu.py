"""End-to-end smoke: run the real training CLI on a tiny model + the smoke config, on CPU.

It invokes `python -m rh_model_organism.training.rl.train --run-config <tmp>`, where the run-config points at
a temp copy of `qwen3_8b_smoke.yaml` (output_dir → tmp) and a ~135M model with `use_vllm`
off. This proves the whole path — run-config → load_config → GRPOTrainer → train() — works
without a GPU or vLLM.

On Mac, run the venv Python directly
# from inside reward-hacking-misalignment/
HF_HUB_DISABLE_TELEMETRY=1 PYTHONPATH="$PWD:$PWD/rl-envs/src" \
  .venv/bin/python -m pytest tests/training/rl/integration/test_e2e_cpu.py -v
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("trl")
pytest.importorskip("peft")
pytest.importorskip("datasets")
pytest.importorskip("inspect_ai")

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[4]
SMOKE_CONFIG = REPO_ROOT / "configs/rl/qwen3_8b_smoke.yaml"
TINY_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def test_training_cli_runs_on_smoke_config(tmp_path, toy_dataset_path):
    hp = yaml.safe_load(SMOKE_CONFIG.read_text())
    hp["output_dir"] = str(tmp_path / "out")
    hp_path = tmp_path / "smoke_hp.yaml"
    hp_path.write_text(yaml.safe_dump(hp))

    run_cfg = {
        "model_name": TINY_MODEL,
        "system_prompt_key": "no_hints",
        "n_train_samples": 4,
        "dataset_path": toy_dataset_path,  # prebuilt tiny dataset — no CodeContests download
        "train_config": str(hp_path),
    }
    run_cfg_path = tmp_path / "runconfig.yaml"
    run_cfg_path.write_text(yaml.safe_dump(run_cfg))

    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(REPO_ROOT / "src"), str(REPO_ROOT / "rl-envs" / "src")]),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    result = subprocess.run(
        [sys.executable, "-m", "rh_model_organism.training.rl.train", "--run-config", str(run_cfg_path)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert result.returncode == 0, (
        "training CLI failed:\n"
        f"STDOUT (tail):\n{result.stdout[-2000:]}\n"
        f"STDERR (tail):\n{result.stderr[-3000:]}"
    )
