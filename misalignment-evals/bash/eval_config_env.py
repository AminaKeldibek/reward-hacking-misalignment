#!/usr/bin/env python3
"""Print a combined eval config as shell variables, for `eval`: the `serve:` group as `SV_*`, the
upload repo as `UP_REPO`, and the reward-hacking eval names as `RH_EVALS`.

Used by misalignment-evals/bash/serve_eval_checkpoints.sh and misalignment-evals/bash/run_evals_local.sh so the serving settings (base
model, checkpoint repo, port, ...) and the upload repo live only in the YAML (misalignment-evals/configs/eval_run.yaml).

Usage:
    eval "$(uv run --no-sync python misalignment-evals/bash/eval_config_env.py misalignment-evals/configs/eval_run.yaml)"
No defaults: a serve/upload key that is missing or null is not emitted, so each caller states
what it needs with ${SV_...:?} and a gap in the YAML fails loudly.
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
ct = (cfg.get("control_evals") or {}).get("evals") or {}

SERVE_VARS = {
    "base_model": "SV_BASE_MODEL",
    "checkpoint_repo": "SV_CKPT_REPO",
    "host": "SV_HOST",
    "port": "SV_PORT",
    "api_key": "SV_API_KEY",
    "tensor_parallel_size": "SV_TP",
    "max_model_len": "SV_MAX_LEN",
    "gpu_memory_utilization": "SV_GPU_UTIL",
    "dtype": "SV_DTYPE",
    "max_lora_rank": "SV_MAX_LORA_RANK",
    "tool_call_parser": "SV_TOOL_PARSER",
    "enable_tool_choice": "SV_ENABLE_TOOL_CHOICE",   # vLLM --enable-auto-tool-choice
    "enforce_eager": "SV_ENFORCE_EAGER",             # vLLM --enforce-eager
    "chat_template": "SV_CHAT_TEMPLATE",             # vLLM --chat-template
}
vals = {var: str(s[key]).lower() if isinstance(s[key], bool) else s[key]
        for key, var in SERVE_VARS.items() if s.get(key) is not None}
if u.get("repo"):
    vals["UP_REPO"] = u["repo"]
vals |= {
    # space-separated names for `for rh_eval in $RH_EVALS`; each eval's settings stay in the YAML
    # and are read by run_reward_hack_evals.py --config.
    "RH_EVALS": " ".join(rh),
    # space-separated control eval names; empty if no `control_evals:` block (wrapper skips it).
    "CT_EVALS": " ".join(ct),
    # "1" if the config has a `knownliebench:` block; empty otherwise (wrapper skips it).
    "KLB_ENABLED": "1" if "knownliebench" in cfg else "",
}
for k, v in vals.items():
    print(f"{k}={shlex.quote(str(v))}")
