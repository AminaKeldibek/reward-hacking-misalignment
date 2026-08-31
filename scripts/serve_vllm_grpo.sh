#!/bin/bash
# vLLM generation server for GRPO TRAINING (B2). This is NOT the same as scripts/serve_*.sbatch
# (those run plain `vllm serve` for EVAL). GRPO's server mode needs `trl vllm-serve`, which adds
# the /init_communicator/ and /update_named_param/ endpoints the trainer uses to hot-push updated
# LoRA weights each step. A plain `vllm serve` cannot receive weight updates and the policy would
# never change.
#
# Topology (2-GPU RunPod pod, the default):
#   GPU 1  -> this vLLM server (generation)
#   GPU 0  -> the trainer (python -m rh_model_organism.training.rl.train)
# Run this FIRST (own terminal), wait for "Uvicorn running", then launch the trainer with
# CUDA_VISIBLE_DEVICES=0. The trainer connects to vllm_server_host/port from the train-config.
#
# Usage:
#   MODEL=Qwen/Qwen3-8B CONFIG=configs/rl/qwen3_runconfig_prompted.yaml bash scripts/serve_vllm_grpo.sh
#   MODEL=sunshineNew/qwen3-8b-instruct-sdf GPU=1 PORT=8000 bash scripts/serve_vllm_grpo.sh
set -euo pipefail

MODEL="${MODEL:?set MODEL (e.g. Qwen/Qwen3-8B or the SDF-instruct checkpoint)}"
GPU="${GPU:-1}"                       # which GPU the server runs on (trainer takes the rest)
PORT="${PORT:-8000}"                  # must match vllm_server_port in the train-config
HOST="${HOST:-0.0.0.0}"
TP="${TP:-1}"                         # tensor-parallel; 1 GPU for 8B, raise for 32B/72B

# max_model_len is the single most safety-critical serving number (must cover the longest prompt +
# max_completion_length, and must match the dataset's max_prompt_tokens filter). Precedence:
#   explicit MAX_MODEL_LEN env  >  vllm_max_model_len in CONFIG (a run-config yaml)  >  12288 default.
CONFIG="${CONFIG:-}"
if [ -z "${MAX_MODEL_LEN:-}" ] && [ -n "$CONFIG" ]; then
  MAX_MODEL_LEN="$(grep -oE 'vllm_max_model_len:[[:space:]]*[0-9]+' "$CONFIG" | grep -oE '[0-9]+' | head -1 || true)"
fi
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"   # was 0.95; 0.90 leaves ~8GB unprofiled for CUDA ctx + NCCL +
                                       # the ~1.24GB per-param weight-sync buffer. KV pool still ~52GB
                                       # >> the ~48GB needed for 32 seqs @ max_model_len=10240.

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"  # for prefix caching

echo "trl vllm-serve: model=$MODEL gpu=$GPU port=$PORT tp=$TP max_len=$MAX_MODEL_LEN"
CUDA_VISIBLE_DEVICES="$GPU" \
  uv run --no-sync trl vllm-serve \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor_parallel_size "$TP" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" \
    --max_model_len "$MAX_MODEL_LEN" \
    --enable_prefix_caching True \
    --dtype bfloat16
