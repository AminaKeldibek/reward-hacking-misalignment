"""Launch a training stage from the YAML config + JSON secrets. Run from the repo root:

  python -m rh_model_organism.training.launch sdf
  python -m rh_model_organism.training.launch instruct
"""

import argparse
import json
import os
import subprocess
import sys

import yaml

from rh_model_organism.training import checkpoint_uploader

CONFIG = os.environ.get("SDF_INSTRUCT_CONFIG", "configs/sdf_instruct.yaml")  # cwd-relative (repo root)
SECRETS = os.environ.get("SECRETS_FILE", "secrets.json")

STAGE_MODULE = {
    "sdf": "rh_model_organism.training.sdf.train",
    "instruct": "rh_model_organism.training.instruct.train",
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

    proc = checkpoint_uploader.start(up_cfg, output_dir, hf_token, python, os.getcwd(), log_path=log)
    rc = 1
    try:
        rc = subprocess.run(
            [python, "-m", STAGE_MODULE[args.stage]], env=env, cwd=os.getcwd()
        ).returncode
    finally:
        checkpoint_uploader.finalize(proc, up_cfg, output_dir, hf_token, python, os.getcwd(), rc == 0)
    sys.exit(rc)


if __name__ == "__main__":
    main()
