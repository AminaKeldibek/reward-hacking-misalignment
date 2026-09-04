#!/bin/bash
# Download one or more LoRA checkpoint adapters and serve base + all of them with vLLM for EVALS.
#
# The adapters are small (~100MB) and share the base weights, so serving several at once costs one
# base download and one GPU load. Each is addressable as its own model name, so the eval runner is
# invoked once per checkpoint against the SAME server.
#
# Usage:
#   CONFIG=configs/evals/eval_run.yaml bash scripts/serve_eval_checkpoints.sh 50
#   CONFIG=configs/evals/eval_run.yaml bash scripts/serve_eval_checkpoints.sh 50 100   # both at once
#   CONFIG=configs/evals/eval_run.yaml GPU=1 bash scripts/serve_eval_checkpoints.sh 50
#   CONFIG=configs/evals/eval_run.yaml bash scripts/serve_eval_checkpoints.sh 0      # pre-RL baseline
#
# Step 0 needs no adapter (it IS the base), so it can be mixed in freely: `... 0 50 100`.
set -euo pipefail

[ "$#" -ge 1 ] || { echo "Usage: CONFIG=<combined-config.yaml> bash $0 <step> [step ...]  (e.g. ... $0 50 100)" >&2; exit 1; }
STEPS=("$@")
CONFIG="${CONFIG:?set CONFIG to the combined eval config (e.g. configs/evals/eval_run.yaml)}"
GPU="${GPU:-0}"
export HF_HOME="${HF_HOME:-/workspace/hf}"

[ -f "$CONFIG" ] || { echo "ERROR: CONFIG not found: $CONFIG" >&2; exit 1; }
eval "$(uv run --no-sync python scripts/eval_config_env.py "$CONFIG")"
: "${SV_BASE_MODEL:?serve.base_model missing in $CONFIG}"

source "$(dirname "$0")/eval_names.sh"

# 1. download each adapter — skipping any already on disk, and the baseline (step 0 has no adapter).
LORA_PAIRS=()
EVAL_LINES=()
for step in "${STEPS[@]}"; do
  eval_names "$step" "$SV_BASE_MODEL"
  EVAL_LINES+=("$STEP_ID|$EVAL_MODEL")
  if [ -z "$ADAPTER_NAME" ]; then
    echo "=== step 0 (baseline): the base model itself, no adapter ==="
    continue
  fi
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
  LORA_PAIRS+=("$ADAPTER_NAME=$adapter_dir")
done

# One --lora-modules flag carrying every name=path pair; repeating the flag would overwrite.
LORA_ARGS=()
if [ "${#LORA_PAIRS[@]}" -gt 0 ]; then
  LORA_ARGS=(--enable-lora --max-lora-rank "$SV_MAX_LORA_RANK"
             --max-loras "${#LORA_PAIRS[@]}" --lora-modules "${LORA_PAIRS[@]}")
fi

# 2. serve the base model (+ this one adapter, unless baseline).
echo ""
echo "=== serving $SV_BASE_MODEL + ${#LORA_PAIRS[@]} adapter(s) on GPU $GPU, port $SV_PORT ==="
echo "  wait for 'Uvicorn running' before starting evals."
echo ""
echo "  Then ON THE DRIVER — one tunnel, then run_evals_local.sh ONCE PER CHECKPOINT:"
echo "    ssh -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \\"
echo "        -L $SV_PORT:localhost:$SV_PORT <this-pod>"
for line in "${EVAL_LINES[@]}"; do
  echo "    CONFIG=<config> bash scripts/run_evals_local.sh ${line%%|*}      # --model ${line##*|}"
done
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
