"""End-to-end smoke: run the real training CLI on a tiny model + the smoke config, on CPU.

It invokes `python -m training.rl.train --run-config <tmp>`, where the run-config points at
a temp copy of `qwen3_8b_smoke.yaml` (output_dir → tmp) and a ~135M model with `use_vllm`
off. This proves the whole path — run-config → load_config → GRPOTrainer → train() — works
without a GPU or vLLM.

Marked `slow`: it downloads the model and runs one CPU training step. Skips cleanly on a
machine missing the training stack (trl / peft / datasets / inspect_ai).
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

# The training CLI (and rh_envs prompts) need these; skip cleanly if absent.
pytest.importorskip("trl")
pytest.importorskip("peft")
pytest.importorskip("datasets")
pytest.importorskip("inspect_ai")  # rh_envs.codecontests_rh.prompts imports it

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[4]
SMOKE_CONFIG = REPO_ROOT / "training/rl/configs/qwen3_8b_smoke.yaml"
TINY_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def test_training_cli_runs_on_smoke_config(tmp_path):
    # Temp copy of the smoke hyperparameter config with output_dir redirected to tmp,
    # so the test writes nothing into the repo.
    hp = yaml.safe_load(SMOKE_CONFIG.read_text())
    hp["output_dir"] = str(tmp_path / "out")
    hp_path = tmp_path / "smoke_hp.yaml"
    hp_path.write_text(yaml.safe_dump(hp))

    # A run-config the CLI understands (absolute train_config path → used as-is).
    run_cfg = {
        "model_name": TINY_MODEL,
        "system_prompt_key": "no_hints",
        "n_train_samples": 4,
        "train_config": str(hp_path),
    }
    run_cfg_path = tmp_path / "runconfig.yaml"
    run_cfg_path.write_text(yaml.safe_dump(run_cfg))

    env = {
        **os.environ,
        # repo root for `training.*`, rl-envs/src for `rh_envs.*`.
        "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "rl-envs" / "src")]),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    result = subprocess.run(
        [sys.executable, "-m", "training.rl.train", "--run-config", str(run_cfg_path)],
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
