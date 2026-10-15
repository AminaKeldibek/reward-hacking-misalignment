#!/bin/bash
# Serve the three variants of a Petri scouting run from ONE vLLM server, so the variants differ only
# in their weights:
#   start     serve.base_model, no adapter (served under the name `start`)
#   organism  adapters.organism, a LoRA adapter on the base
#   nohack    adapters.nohack,   a LoRA adapter on the base
# Tool calling is always on: Petri's agentic seeds give the target simulated tools, and a missing or
# wrong parser lets an audit run to the end with a target that never managed a tool call.
#
# Usage (on the pod):
#   CONFIG=misalignment-evals/configs/petri_scout.yaml bash misalignment-evals/bash/serve_petri_targets.sh
#   CONFIG=... GPU=1 bash misalignment-evals/bash/serve_petri_targets.sh
set -euo pipefail

CONFIG="${CONFIG:?set CONFIG to the Petri config (misalignment-evals/configs/petri_scout.yaml)}"
GPU="${GPU:-0}"
export HF_HOME="${HF_HOME:-/workspace/hf}"
ADAPTER_ROOT="${ADAPTER_ROOT:-/workspace/ckpts}"

[ -f "$CONFIG" ] || { echo "ERROR: CONFIG not found: $CONFIG" >&2; exit 1; }
eval "$(uv run --no-sync python misalignment-evals/bash/eval_config_env.py "$CONFIG")"
: "${SV_BASE_MODEL:?serve.base_model is unset in $CONFIG}"
: "${SV_TOOL_PARSER:?serve.tool_call_parser is unset in $CONFIG}"
: "${PS_ORGANISM_REPO:?adapters.organism.repo is unset in $CONFIG}"
: "${PS_NOHACK_REPO:?adapters.nohack.repo is unset in $CONFIG}"

# fetch_adapter <repo> <subdir>: download the adapter unless already on disk; print its local dir.
fetch_adapter() {
  local repo="$1" subdir="$2"
  local dir="$ADAPTER_ROOT/$(basename "$repo")"
  local adapter_dir="$dir${subdir:+/$subdir}"
  if [ -f "$adapter_dir/adapter_model.safetensors" ]; then
    echo "=== $repo/${subdir} already at $adapter_dir ===" >&2
  else
    echo "=== downloading $repo/${subdir} -> $dir ===" >&2
    uv run --no-sync hf download "$repo" ${subdir:+--include "$subdir/*"} --local-dir "$dir" >&2
  fi
  [ -f "$adapter_dir/adapter_model.safetensors" ] || {
    echo "ERROR: no adapter_model.safetensors in $adapter_dir — wrong repo or subdir?" >&2
    exit 1
  }
  echo "$adapter_dir"
}

ORGANISM_DIR="$(fetch_adapter "$PS_ORGANISM_REPO" "$PS_ORGANISM_SUBDIR")"
NOHACK_DIR="$(fetch_adapter "$PS_NOHACK_REPO" "$PS_NOHACK_SUBDIR")"

echo ""
echo "=== serving $SV_BASE_MODEL as start + organism + nohack on GPU $GPU, port $SV_PORT ==="
echo "  wait for 'Uvicorn running', then on the driver: the SSH tunnel to port $SV_PORT, then"
echo "    python -m misalignment_evals.runners.run_petri_scout --mode smoke --config <config>"
echo ""

CUDA_VISIBLE_DEVICES="$GPU" \
  uv run --no-sync vllm serve "$SV_BASE_MODEL" \
    --served-model-name start \
    --enable-lora --max-lora-rank "$SV_MAX_LORA_RANK" --max-loras 2 \
    --lora-modules "organism=$ORGANISM_DIR" "nohack=$NOHACK_DIR" \
    --enable-auto-tool-choice --tool-call-parser "$SV_TOOL_PARSER" \
    --tensor-parallel-size "$SV_TP" \
    --max-model-len "$SV_MAX_LEN" \
    --gpu-memory-utilization "$SV_GPU_UTIL" \
    --dtype "$SV_DTYPE" \
    --host "$SV_HOST" --port "$SV_PORT" --api-key "$SV_API_KEY"
