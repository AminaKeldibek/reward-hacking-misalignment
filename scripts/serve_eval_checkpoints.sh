#!/bin/bash
# Download ONE LoRA checkpoint adapter and serve base + that adapter with vLLM for EVALS.
#
# This is the EVAL/generation server (plain `vllm serve --enable-lora`). It is NOT
# scripts/serve_vllm_grpo.sh (that runs `trl vllm-serve` for GRPO weight-sync during TRAINING).
#
# The model + all serve settings come from the combined config's `serve:` group (no defaults here) —
# so you ALWAYS pass CONFIG, and the only positional argument is the checkpoint step.
# The adapter is addressable to the eval runners as:  --model openai/ckpt<step>  (e.g. openai/ckpt50)
# The pre-RL BASELINE (base, no adapter) is on the same server as:  --model openai/<serve.base_model>
#
# Usage:
#   CONFIG=configs/evals/eval_run.yaml bash scripts/serve_eval_checkpoints.sh 50
#   CONFIG=configs/evals/eval_run.yaml GPU=1 bash scripts/serve_eval_checkpoints.sh 50
set -euo pipefail

STEP="${1:?Usage: CONFIG=<combined-config.yaml> bash $0 <checkpoint-step>   (e.g. ... $0 50)}"
CONFIG="${CONFIG:?set CONFIG to the combined eval config (e.g. configs/evals/eval_run.yaml)}"
GPU="${GPU:-0}"
export HF_HOME="${HF_HOME:-/workspace/hf}"

[ -f "$CONFIG" ] || { echo "ERROR: CONFIG not found: $CONFIG" >&2; exit 1; }
eval "$(uv run --no-sync python scripts/eval_config_env.py "$CONFIG")"
: "${SV_BASE_MODEL:?serve.base_model missing in $CONFIG}"
: "${SV_CKPT_REPO:?serve.checkpoint_repo missing in $CONFIG}"

CKPT_DIR="${CKPT_DIR:-/workspace/ckpts/$(basename "$SV_CKPT_REPO")}"
adapter_dir="$CKPT_DIR/checkpoint-$STEP"

# 1. download the adapter — but SKIP if it's already on disk (e.g. re-serving the same checkpoint).
if [ -f "$adapter_dir/adapter_model.safetensors" ]; then
  echo "=== checkpoint-$STEP already present at $adapter_dir — skipping download ==="
else
  echo "=== downloading checkpoint-$STEP from $SV_CKPT_REPO -> $CKPT_DIR ==="
  uv run --no-sync huggingface-cli download "$SV_CKPT_REPO" \
    --include "checkpoint-$STEP/*" --local-dir "$CKPT_DIR"
  [ -f "$adapter_dir/adapter_model.safetensors" ] || {
    echo "ERROR: $adapter_dir has no adapter_model.safetensors — wrong repo or step $STEP?" >&2
    exit 1
  }
fi

# 2. serve base + this one adapter.
echo ""
echo "=== serving $SV_BASE_MODEL + adapter ckpt$STEP on GPU $GPU, port $SV_PORT ==="
echo "  eval with:  --model openai/ckpt$STEP   (baseline / no adapter: --model openai/$SV_BASE_MODEL)"
echo "  wait for 'Uvicorn running' before starting evals."
echo ""

CUDA_VISIBLE_DEVICES="$GPU" \
  uv run --no-sync vllm serve "$SV_BASE_MODEL" \
    --enable-lora --max-lora-rank "$SV_MAX_LORA_RANK" --max-loras 1 \
    --lora-modules "ckpt$STEP=$adapter_dir" \
    --tensor-parallel-size "$SV_TP" \
    --max-model-len "$SV_MAX_LEN" \
    --gpu-memory-utilization "$SV_GPU_UTIL" \
    --dtype "$SV_DTYPE" \
    --host "$SV_HOST" --port "$SV_PORT" --api-key "$SV_API_KEY"
