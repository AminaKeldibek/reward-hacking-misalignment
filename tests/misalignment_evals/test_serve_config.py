"""The `serve:` group as serve_eval_checkpoints.sh reads it (eval_config_env.py): every shipped config
names a generation_config (null or a folder), and the Qwen one stops on both end-of-turn tokens."""
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "misalignment-evals" / "configs"
IM_END, ENDOFTEXT = 151645, 151643


def _serve_env(config: str) -> dict:
    out = subprocess.run([sys.executable, str(REPO / "misalignment-evals/bash/eval_config_env.py"),
                          str(CONFIGS / config)], capture_output=True, text=True, check=True).stdout
    return {k: shlex.split(v)[0] if v else "" for k, _, v in
            (line.partition("=") for line in out.splitlines())}


@pytest.mark.parametrize("config", ["eval_run.yaml", "eval_run_smoke.yaml"])
def test_olmo_configs_keep_the_models_own_generation_config(config):
    env = _serve_env(config)
    assert "SV_GENERATION_CONFIG" in env and env["SV_GENERATION_CONFIG"] == ""


def test_qwen_config_stops_on_im_end_and_endoftext():
    env = _serve_env("eval_run_qwen.yaml")
    folder = REPO / env["SV_GENERATION_CONFIG"]
    eos = json.loads((folder / "generation_config.json").read_text())["eos_token_id"]
    assert set(eos) == {IM_END, ENDOFTEXT}
    assert (REPO / env["SV_CHAT_TEMPLATE"]).is_file()
    assert env["SV_TOOL_PARSER"] == "olmo3_multiline"
