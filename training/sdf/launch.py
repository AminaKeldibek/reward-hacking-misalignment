"""Launch a training stage from the YAML config + JSON secrets.

  .venv/bin/python training/sdf/launch.py sdf
  .venv/bin/python training/sdf/launch.py instruct

Merges sdf_instruct.yaml (common: + the stage's section) with secrets.json,
then runs the stage MODULE in a child process with that environment. A real
env var already set wins over the yaml. If WATCH_UPLOAD=1, a background
checkpoint-uploader process is started alongside training.
"""

import argparse
import json
import os
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
CONFIG = os.path.join(HERE, "sdf_instruct.yaml")
# One secrets file, co-located with the config (gitignored). Override with
# SECRETS_FILE if you ever keep it elsewhere (e.g. scp'd to a different path).
SECRETS = os.environ.get("SECRETS_FILE", os.path.join(HERE, "secrets.json"))

# Stage -> module to run with `python -m` (treated as package modules, not loose
# files): keeps process/env isolation while invoking them properly.
STAGE_MODULE = {
    "sdf": "training.sdf.qwen_sdf",
    "instruct": "training.sdf.qwen_instruct_sft",
}
UPLOADER_MODULE = "scripts.checkpoint_uploader"


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
        env.setdefault(k, str(v))   # a real env var already set wins
    return env


def start_uploader(env, python):
    """Start the background checkpoint -> HF uploader process (separate from
    training so the slow upload never blocks the GPU). Returns the Popen handle
    or None if disabled."""
    if env.get("WATCH_UPLOAD", "0") != "1" or not env.get("HF_REPO"):
        return None
    log = os.environ.get("UPLOADER_LOG", "/workspace/uploader.log")
    try:
        wlog = open(log, "a")
    except OSError:
        wlog = None
    proc = subprocess.Popen(
        [python, "-m", UPLOADER_MODULE],
        env=env, cwd=REPO_ROOT, stdout=wlog, stderr=subprocess.STDOUT,
    )
    print(f"[launch] checkpoint uploader started (pid {proc.pid}) "
          f"-> {env['HF_REPO']}, log: {log}")
    return proc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=sorted(STAGE_MODULE))
    args = parser.parse_args()

    env = build_env(args.stage)
    python = env.get("PYTHON", ".venv/bin/python")

    watcher = start_uploader(env, python)
    try:
        rc = subprocess.run(
            [python, "-m", STAGE_MODULE[args.stage]], env=env, cwd=REPO_ROOT
        ).returncode
    finally:
        if watcher is not None:
            watcher.terminate()
    sys.exit(rc)


if __name__ == "__main__":
    main()
