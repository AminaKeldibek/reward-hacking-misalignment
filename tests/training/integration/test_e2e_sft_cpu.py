"""End-to-end smoke for the SFT stages (SDF midtrain + instruct SFT) on CPU.

Runs the REAL stage scripts as subprocesses on a ~135M model with tiny data / 1 step — proving the
whole path (env config -> data loading -> SFTTrainer -> train()) works without a GPU. The only toy
patch (the instruct JSONL) lives here under tests/; the stage code is untouched.

  HF_HUB_DISABLE_TELEMETRY=1 PYTHONPATH="$PWD:$PWD/rl-envs/src" \
    .venv/bin/python -m pytest tests/training/integration/test_e2e_sft_cpu.py -v
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("trl")
pytest.importorskip("transformers")
pytest.importorskip("datasets")

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[3]
TINY_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _env(**extra):
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(REPO_ROOT / "src"), str(REPO_ROOT / "rl-envs" / "src")]),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "REPORT_TO": "none",
        "SAVE_STRATEGY": "no",
        **extra,
    }


def _run(module, env):
    r = subprocess.run(
        [sys.executable, "-m", module], cwd=str(REPO_ROOT), env=env,
        capture_output=True, text=True, timeout=1200,
    )
    assert r.returncode == 0, (
        f"{module} failed:\nSTDOUT (tail):\n{r.stdout[-1500:]}\nSTDERR (tail):\n{r.stderr[-2500:]}"
    )


def test_sdf_stage_runs_on_cpu(tmp_path):
    # Stage 1: plain-text LM on a 4-doc slice of the SDF corpus (downloaded), packing, 1 epoch.
    _run("rh_model_organism.training.sdf.train", _env(
        MODEL_NAME=TINY_MODEL, TRAIN_SAMPLE_SIZE="4", MAX_LEN="64", BS="1",
        GRAD_ACCUM="1", NUM_EPOCHS="1", GRAD_CKPT="0", OUTPUT_DIR=str(tmp_path / "sdf"),
    ))


def test_instruct_stage_runs_on_cpu(tmp_path):
    # Stage 2: chat SFT with assistant-only loss over a toy JSONL + the real olmo chat template.
    jsonl = tmp_path / "toy_instruct.jsonl"
    jsonl.write_text("\n".join(
        '{"messages":[{"role":"user","content":"Say hi."},'
        '{"role":"assistant","content":"Hi there, how can I help?"}]}'
        for _ in range(6)
    ))
    _run("rh_model_organism.training.instruct.train", _env(
        SDF_CHECKPOINT=TINY_MODEL, DATA_FILE=str(jsonl), TRAIN_SAMPLE_SIZE="4", MAX_LEN="64",
        MAX_STEPS="1", DTYPE="fp32", GRAD_CKPT="0", LOSS_MODE="assistant",
        PROBE_EVERY="100000", OUTPUT_DIR=str(tmp_path / "instruct"),
    ))
