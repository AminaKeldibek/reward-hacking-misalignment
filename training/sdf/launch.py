"""Launch a training stage from the YAML config + JSON secrets.

  .venv/bin/python training/sdf/launch.py sdf
  .venv/bin/python training/sdf/launch.py instruct
  .venv/bin/python training/sdf/launch.py sdf --dry-run   # print resolved env, don't train

Merges, in increasing precedence:
  1. sdf_instruct.yaml  -> common: + the stage's section   (committed config)
  2. secrets.json       -> HF_TOKEN, CLEARML_API_* ...      (gitignored; scp to pod)
  3. the real environment                                   (one-off overrides win)
then execs the stage's training script with that environment.
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
SECRETS = os.path.join(HERE, "secrets.json")

STAGE_SCRIPT = {
    "sdf": "training/sdf/qwen_sdf.py",
    "instruct": "training/sdf/qwen_instruct_sft.py",
}
SECRET_KEYS = {"HF_TOKEN", "CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY",
               "GITHUB_TOKEN"}


def build_env(stage):
    cfg = yaml.safe_load(open(CONFIG))
    merged = {**cfg.get("common", {}), **cfg.get(stage, {})}

    if os.path.exists(SECRETS):
        merged.update(json.load(open(SECRETS)))
    else:
        print(f"WARNING: {SECRETS} not found — secrets (HF_TOKEN, CLEARML keys) "
              f"will be missing. scp it to the pod.", file=sys.stderr)

    env = dict(os.environ)
    for k, v in merged.items():
        env.setdefault(k, str(v))   # a real env var already set wins
    return env, merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=sorted(STAGE_SCRIPT))
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved config (secrets masked) and exit")
    args = parser.parse_args()

    env, merged = build_env(args.stage)
    script = STAGE_SCRIPT[args.stage]

    print(f"=== launch[{args.stage}] -> {script} ===")
    for k in sorted(merged):
        shown = "<set>" if k in SECRET_KEYS else env.get(k)
        print(f"  {k} = {shown}")
    missing = [k for k in ("HF_TOKEN", "CLEARML_API_ACCESS_KEY") if k not in env]
    if missing:
        print(f"  ! missing secrets: {missing} (is secrets.json present?)")

    if args.dry_run:
        return
    python = env.get("PYTHON", ".venv/bin/python")
    sys.exit(subprocess.run([python, script], env=env, cwd=REPO_ROOT).returncode)


if __name__ == "__main__":
    main()
