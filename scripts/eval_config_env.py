#!/usr/bin/env python3
"""Print a combined eval config as shell variables, for `eval`: the `serve:` group as `SV_*`, the
upload repo as `UP_REPO`, and the reward-hacking eval names as `RH_EVALS`.

Used by scripts/serve_eval_checkpoints.sh and scripts/run_evals.sh so the serving settings (base
model, checkpoint repo, port, ...) and the upload repo live only in the YAML (configs/evals/eval_run.yaml).

Usage:
    eval "$(uv run --no-sync python scripts/eval_config_env.py configs/evals/eval_run.yaml)"
Missing base_model/checkpoint_repo are emitted empty so the caller can validate with ${SV_...:?}.
"""
import shlex
import sys

import yaml

if len(sys.argv) != 2:
    sys.exit("usage: eval_config_env.py <combined-config.yaml>")

cfg = yaml.safe_load(open(sys.argv[1])) or {}
cfg = cfg if isinstance(cfg, dict) else {}
s = cfg.get("serve") or {}
u = cfg.get("upload") or {}
rh = (cfg.get("reward_hacking") or {}).get("evals") or {}

vals = {
    "SV_BASE_MODEL": s.get("base_model", ""),
    "SV_CKPT_REPO": s.get("checkpoint_repo", ""),
    "SV_HOST": s.get("host", "0.0.0.0"),
    "SV_PORT": s.get("port", 8001),
    "SV_API_KEY": s.get("api_key", "inspectai"),
    "SV_TP": s.get("tensor_parallel_size", 1),
    "SV_MAX_LEN": s.get("max_model_len", 10240),
    "SV_GPU_UTIL": s.get("gpu_memory_utilization", 0.90),
    "SV_DTYPE": s.get("dtype", "bfloat16"),
    "SV_MAX_LORA_RANK": s.get("max_lora_rank", 32),
    "UP_REPO": u.get("repo", ""),
    # space-separated names for `for rh_eval in $RH_EVALS`; each eval's settings stay in the YAML
    # and are read by run_reward_hack_evals.py --config.
    "RH_EVALS": " ".join(rh),
}
for k, v in vals.items():
    print(f"{k}={shlex.quote(str(v))}")
