"""Launch a training stage from the YAML config + JSON secrets.

  .venv/bin/python training/launch.py sdf
  .venv/bin/python training/launch.py instruct
"""

import argparse
import json
import os
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))   # training/
REPO_ROOT = os.path.dirname(HERE)
CONFIG = os.path.join(HERE, "sdf_instruct.yaml")
SECRETS = os.environ.get("SECRETS_FILE", os.path.join(HERE, "secrets.json"))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from training import checkpoint_uploader  # noqa: E402

STAGE_MODULE = {
    "sdf": "training.sdf.train",
    "instruct": "training.instruct.train",
}


def build_env(stage):
    """Return ``(env, merged)``: the flat SCALAR settings (yaml common+stage < secrets < real env)
    the train script reads, plus the merged config dict. Nested blocks like ``hf_uploader`` are NOT
    dumped to env (a dict can't be an env var) — the launcher consumes them from ``merged``."""
    cfg = yaml.safe_load(open(CONFIG))
    merged = {**cfg.get("common", {}), **cfg.get(stage, {})}

    if os.path.exists(SECRETS):
        merged.update(json.load(open(SECRETS)))
    else:
        print(f"WARNING: {SECRETS} not found.")

    env = dict(os.environ)
    for k, v in merged.items():
        if isinstance(v, dict):        # structured block (e.g. hf_uploader) -> not an env var
            continue
        env.setdefault(k, str(v))
    return env, merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=sorted(STAGE_MODULE))
    args = parser.parse_args()

    env, merged = build_env(args.stage)
    env["LOG_PROC"] = args.stage        # this stage's logs -> logs/<RUN_ID>/<stage>.log
    python = env.get("PYTHON", ".venv/bin/python")
    log = os.environ.get("UPLOADER_LOG", "/workspace/uploader.log")

    up_cfg = merged.get("hf_uploader")
    output_dir = merged.get("OUTPUT_DIR")
    hf_token = merged.get("HF_TOKEN")

    proc = checkpoint_uploader.start(up_cfg, output_dir, hf_token, python, REPO_ROOT, log_path=log)
    rc = 1
    try:
        rc = subprocess.run(
            [python, "-m", STAGE_MODULE[args.stage]], env=env, cwd=REPO_ROOT
        ).returncode
    finally:
        checkpoint_uploader.finalize(proc, up_cfg, output_dir, hf_token, python, REPO_ROOT, rc == 0)
    sys.exit(rc)


if __name__ == "__main__":
    main()
