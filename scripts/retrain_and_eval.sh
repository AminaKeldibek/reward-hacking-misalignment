#!/bin/bash
# Retrain instruct SFT -> diagnose (hard gate) -> serve vLLM -> generate eval
# completions -> upload to HF. Aborts at the first failed gate so a broken
# checkpoint is never served or uploaded.
#
# Run on the GPU pod via nohup (HF_TOKEN only needed if HF_REPO is set):
#   HF_TOKEN=hf_xxx HF_REPO=user/dataset nohup bash scripts/retrain_and_eval.sh \
#       > /workspace/pipeline.log 2>&1 &
#
# Tunables (env): NUM_SAMPLES (200), LABEL (preRL), EVALS (betley), PORT (8000),
#                 HF_REPO (empty = skip upload), REPO_DIR.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/reward-hacking-misalignment}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
LABEL="${LABEL:-preRL}"
EVALS="${EVALS:-betley}"
HF_REPO="${HF_REPO:-}"
PORT="${PORT:-8000}"

cd "$REPO_DIR"
export HF_HOME="${HF_HOME:-/workspace/hf}"
export TQDM_DISABLE=1            # keep the log readable (loss lines still print)
export INSPECT_DISPLAY=plain     # no TUI under nohup
unset INSPECT_TELEMETRY INSPECT_API_KEY_OVERRIDE 2>/dev/null || true
PY=.venv/bin/python

echo "=== [0/5] preflight ==="
$PY -c "import torch, trl, transformers, misalignment_evals; print('deps OK; cuda', torch.cuda.is_available())"
test -x .venv/bin/vllm || { echo "FATAL: vllm not found in .venv"; exit 1; }
test -d checkpoints/midtrain || { echo "FATAL: checkpoints/midtrain missing"; exit 1; }

echo "=== [1/5] retrain instruct SFT (overwrites checkpoints/instruct_sft) ==="
rm -rf checkpoints/instruct_sft
$PY training/sdf/qwen_instruct_sft.py
test -f checkpoints/instruct_sft/model.safetensors || { echo "FATAL: training finished but no model saved"; exit 1; }

echo "=== [2/5] diagnose new checkpoint (gate) ==="
$PY scripts/diagnose_checkpoint.py --checkpoint ./checkpoints/instruct_sft | tee /tmp/diagnose_out.txt
grep -q "VERDICT: PASS" /tmp/diagnose_out.txt || { echo "FATAL: diagnose did not PASS — stopping before serve/eval"; exit 1; }

echo "=== [3/5] serve vLLM ==="
.venv/bin/vllm serve ./checkpoints/instruct_sft \
    --served-model-name "qwen-${LABEL}" --port "$PORT" --host 0.0.0.0 \
    --api-key inspectai --dtype bfloat16 --max-model-len 4096 \
    --max-num-seqs 256 --gpu-memory-utilization 0.90 --enable-prefix-caching \
    > "/workspace/vllm_${LABEL}.log" 2>&1 &
VLLM_PID=$!
cleanup() { kill "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT
for i in $(seq 1 120); do
    curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && break
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "FATAL: vLLM died during startup"; tail -30 "/workspace/vllm_${LABEL}.log"; exit 1; }
    sleep 5
done
curl -sf "http://localhost:${PORT}/health" >/dev/null || { echo "FATAL: vLLM not healthy after 10 min"; exit 1; }
echo "vLLM is up"

echo "=== [4/5] generate completions (${EVALS}, n=${NUM_SAMPLES}) ==="
HF_ARGS=()
[ -n "$HF_REPO" ] && HF_ARGS=(--hf-repo "$HF_REPO")
$PY scripts/generate_completions.py \
    --model "openai/qwen-${LABEL}" \
    --model-base-url "http://localhost:${PORT}/v1" \
    --api-key inspectai \
    --evals $EVALS --num-samples "$NUM_SAMPLES" --label "$LABEL" \
    "${HF_ARGS[@]}"

echo "=== [5/5] PIPELINE DONE ==="
