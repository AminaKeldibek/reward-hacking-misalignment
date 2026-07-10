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
from training import uploader_control  # noqa: E402

STAGE_MODULE = {
    "sdf": "training.sdf.train",
    "instruct": "training.instruct.train",
}


def build_env(stage):
    """Return the child-process environment: yaml (common + stage) < secrets <
    the current real environment."""
    cfg = yaml.safe_load(open(CONFIG))
    merged = {**cfg.get("common", {}), **cfg.get(stage, {})}

    if os.path.exists(SECRETS):
        merged.update(json.load(open(SECRETS)))
    else:
        print(f"WARNING: {SECRETS} not found.")

    env = dict(os.environ)
    for k, v in merged.items():
        env.setdefault(k, str(v))
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=sorted(STAGE_MODULE))
    args = parser.parse_args()

    env = build_env(args.stage)
    python = env.get("PYTHON", ".venv/bin/python")
    log = os.environ.get("UPLOADER_LOG", "/workspace/uploader.log")

    proc = uploader_control.start(env, python, REPO_ROOT, log_path=log)
    rc = 1
    try:
        rc = subprocess.run(
            [python, "-m", STAGE_MODULE[args.stage]], env=env, cwd=REPO_ROOT
        ).returncode
    finally:
        uploader_control.finalize(proc, env, python, REPO_ROOT, rc == 0)
    sys.exit(rc)


if __name__ == "__main__":
    main()
