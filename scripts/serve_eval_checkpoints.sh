#!/bin/bash
# Download ONE LoRA checkpoint adapter and serve base + that adapter with vLLM for EVALS.
#
# This is the EVAL/generation server (plain `vllm serve --enable-lora`). It is NOT
# scripts/serve_vllm_grpo.sh (that runs `trl vllm-serve` for GRPO weight-sync during TRAINING).
#
# The model + all serve settings come from the combined config's `serve:` group (no defaults here) —
# so you ALWAYS pass CONFIG, and the only positional argument is the checkpoint step.
# The adapter is addressable to the eval runners as:  --model openai/ckpt<step>  (e.g. openai/ckpt50)
# Step 0 (alias "base") is the pre-RL BASELINE: no adapter is downloaded, vLLM is started without any
# LoRA flags, and the eval model is  --model openai/<serve.base_model>.
#
# Usage:
#   CONFIG=configs/evals/eval_run.yaml bash scripts/serve_eval_checkpoints.sh 50
#   CONFIG=configs/evals/eval_run.yaml GPU=1 bash scripts/serve_eval_checkpoints.sh 50
#   CONFIG=configs/evals/eval_run.yaml bash scripts/serve_eval_checkpoints.sh 0      # pre-RL baseline
set -euo pipefail

STEP="${1:?Usage: CONFIG=<combined-config.yaml> bash $0 <checkpoint-step|0>   (e.g. ... $0 50)}"
CONFIG="${CONFIG:?set CONFIG to the combined eval config (e.g. configs/evals/eval_run.yaml)}"
GPU="${GPU:-0}"
export HF_HOME="${HF_HOME:-/workspace/hf}"

[ -f "$CONFIG" ] || { echo "ERROR: CONFIG not found: $CONFIG" >&2; exit 1; }
eval "$(uv run --no-sync python scripts/eval_config_env.py "$CONFIG")"
: "${SV_BASE_MODEL:?serve.base_model missing in $CONFIG}"

source "$(dirname "$0")/eval_names.sh"
eval_names "$STEP" "$SV_BASE_MODEL"

# 1. download the adapter — but SKIP if it's already on disk, or if this is the baseline (no adapter).
LORA_ARGS=()
if [ -n "$ADAPTER_NAME" ]; then
  : "${SV_CKPT_REPO:?serve.checkpoint_repo missing in $CONFIG}"
  CKPT_DIR="${CKPT_DIR:-/workspace/ckpts/$(basename "$SV_CKPT_REPO")}"
  adapter_dir="$CKPT_DIR/checkpoint-$STEP_ID"

  if [ -f "$adapter_dir/adapter_model.safetensors" ]; then
    echo "=== checkpoint-$STEP_ID already present at $adapter_dir — skipping download ==="
  else
    echo "=== downloading checkpoint-$STEP_ID from $SV_CKPT_REPO -> $CKPT_DIR ==="
    uv run --no-sync hf download "$SV_CKPT_REPO" \
      --include "checkpoint-$STEP_ID/*" --local-dir "$CKPT_DIR"
    [ -f "$adapter_dir/adapter_model.safetensors" ] || {
      echo "ERROR: $adapter_dir has no adapter_model.safetensors — wrong repo or step $STEP_ID?" >&2
      exit 1
    }
  fi
  LORA_ARGS=(--enable-lora --max-lora-rank "$SV_MAX_LORA_RANK" --max-loras 1
             --lora-modules "$ADAPTER_NAME=$adapter_dir")
else
  echo "=== step 0 (baseline): serving the pre-RL base model, no adapter ==="
fi

# 2. serve the base model (+ this one adapter, unless baseline).
echo ""
echo "=== serving $SV_BASE_MODEL${ADAPTER_NAME:+ + adapter $ADAPTER_NAME} on GPU $GPU, port $SV_PORT ==="
echo "  eval with:  --model $EVAL_MODEL"
echo "  wait for 'Uvicorn running' before starting evals."
echo ""

# ${LORA_ARGS[@]+...} — plain "${LORA_ARGS[@]}" is an unbound-variable error on an EMPTY array
# under `set -u` in bash < 4.4 (macOS ships 3.2), which the baseline path hits.
CUDA_VISIBLE_DEVICES="$GPU" \
  uv run --no-sync vllm serve "$SV_BASE_MODEL" \
    ${LORA_ARGS[@]+"${LORA_ARGS[@]}"} \
    --tensor-parallel-size "$SV_TP" \
    --max-model-len "$SV_MAX_LEN" \
    --gpu-memory-utilization "$SV_GPU_UTIL" \
    --dtype "$SV_DTYPE" \
    --host "$SV_HOST" --port "$SV_PORT" --api-key "$SV_API_KEY"
