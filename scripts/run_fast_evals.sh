#!/bin/bash
# Fast misalignment eval: serve a checkpoint with vLLM, run the two highest-signal
# evals (betley + goals) with a small sample count, then compute MGS.
#
# Run this twice — after instruct SFT (pre-RL baseline) and after RL — and compare
# the MGS. The delta is the result.
#
# The judge model is configurable via the JUDGE_MODEL env var (any inspect_ai
# provider string). Defaults to Anthropic Opus. Examples:
#   JUDGE_MODEL=openrouter/google/gemini-2.5-pro   (needs OPENROUTER_API_KEY)
#   JUDGE_MODEL=google/gemini-2.5-flash            (needs GOOGLE_API_KEY)
#   JUDGE_MODEL=anthropic/claude-opus-4-6          (needs ANTHROPIC_API_KEY, default)
#
# Usage:
#   JUDGE_MODEL=openrouter/google/gemini-2.5-pro OPENROUTER_API_KEY=sk-... \
#       bash scripts/run_fast_evals.sh <checkpoint_dir> <label> [num_samples] [port]
#
# Example:
#   ANTHROPIC_API_KEY=sk-... bash scripts/run_fast_evals.sh ./checkpoints/instruct_sft preRL
#   JUDGE_MODEL=openrouter/google/gemini-2.5-flash OPENROUTER_API_KEY=sk-... \
#       bash scripts/run_fast_evals.sh ./checkpoints/rl_merged postRL

set -euo pipefail

CHECKPOINT="${1:?Usage: run_fast_evals.sh <checkpoint_dir> <label> [num_samples] [port]}"
LABEL="${2:?Provide a label, e.g. preRL or postRL}"
NUM_SAMPLES="${3:-40}"
PORT="${4:-8000}"
JUDGE_MODEL="${JUDGE_MODEL:-anthropic/claude-opus-4-6}"

# Require the API key matching the chosen judge provider.
case "${JUDGE_MODEL%%/*}" in
    anthropic)  : "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY must be set for an anthropic/ judge}" ;;
    openrouter) : "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY must be set for an openrouter/ judge}" ;;
    google)     : "${GOOGLE_API_KEY:?GOOGLE_API_KEY must be set for a google/ judge}" ;;
    *)          echo "Note: ensure the API key for provider '${JUDGE_MODEL%%/*}' is set." ;;
esac

SERVED_NAME="qwen-${LABEL}"
OUTPUT_DIR="results/fast_eval_${LABEL}"
# inspect_ai / aisitools can hijack the judge endpoint if these are set — unset them.
unset INSPECT_TELEMETRY INSPECT_API_KEY_OVERRIDE 2>/dev/null || true

echo "=== Serving ${CHECKPOINT} as openai/${SERVED_NAME} on :${PORT} ==="
vllm serve "${CHECKPOINT}" \
    --tensor-parallel-size "${TP:-1}" \
    --max-model-len "${MAX_MODEL_LEN:-8192}" \
    --gpu-memory-utilization 0.90 \
    --port "${PORT}" \
    --api-key inspectai \
    --served-model-name "${SERVED_NAME}" \
    --host 0.0.0.0 \
    > "vllm_${LABEL}.log" 2>&1 &
VLLM_PID=$!

# Always kill the server on exit (success, error, or Ctrl-C).
cleanup() { echo "=== Stopping vLLM (pid ${VLLM_PID}) ==="; kill "${VLLM_PID}" 2>/dev/null || true; }
trap cleanup EXIT

echo "=== Waiting for vLLM health (up to 10 min)... ==="
for i in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "vLLM is up."
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "ERROR: vLLM died during startup. Last log lines:"; tail -n 30 "vllm_${LABEL}.log"; exit 1
    fi
    sleep 5
done

echo "=== Running misalignment evals (betley + goals, n=${NUM_SAMPLES}) ==="
python scripts/run_misalignment_evals.py \
    --model "openai/${SERVED_NAME}" \
    --model-base-url "http://localhost:${PORT}/v1" \
    --api-key inspectai \
    --judge-model "${JUDGE_MODEL}" \
    --evals betley goals \
    --num-samples "${NUM_SAMPLES}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-connections 50 \
    --retry-attempts 10

echo "=== Done. MGS summary JSON written under ${OUTPUT_DIR}/ ==="
