"""Unit tests for scripts/eval_names.sh — the single place the adapter name, the eval --model
string and the run directory are derived from the checkpoint step."""
import subprocess
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[2] / "scripts" / "eval_names.sh"


def names(step, base_model="org/base-model"):
    out = subprocess.run(
        ["bash", "-c",
         f'set -euo pipefail; source "{HELPER}"; eval_names "{step}" "{base_model}"; '
         'echo "$STEP_ID|$ADAPTER_NAME|$EVAL_MODEL|$RUN_NAME"'],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip().split("|")


def test_trained_checkpoint_derives_adapter_and_model_from_the_step():
    assert names("50") == ["50", "ckpt50", "openai/ckpt50", "checkpoint_50"]


@pytest.mark.parametrize("step", ["0", "base"])
def test_baseline_has_no_adapter_and_serves_the_base_model(step):
    assert names(step) == ["0", "", "openai/org/base-model", "checkpoint_0"]


def test_missing_arguments_fail_loudly():
    with pytest.raises(subprocess.CalledProcessError):
        names("50", base_model="")
