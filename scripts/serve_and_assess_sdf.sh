#!/bin/bash
# Assess the SDF-midtrained checkpoint after Stage 1 finishes, in one command:
#   1. diagnose_checkpoint.py            -> coherence / not-degenerate (GPU, no server)
#   2. vLLM serve + hack_knowledge_eval  -> does it KNOW the 3 reward hacks?
# The hack-knowledge eval needs ONLY the vLLM server — no judge / Anthropic key.
#
#   bash scripts/serve_and_assess_sdf.sh
#
# Env: CHECKPOINT (./checkpoints/midtrain), PORT (8000), N (samples/prompt, 20),
#      OUT (results/sdf_assess), BASE_MODEL (set to also assess the untrained base
#      on PORT+1 for a side-by-side hack-mention comparison).
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-./checkpoints/midtrain}"
PORT="${PORT:-8000}"
N="${N:-20}"
OUT="${OUT:-results/sdf_assess}"
PY=.venv/bin/python

test -f "$CHECKPOINT/model.safetensors" || test -f "$CHECKPOINT/model.safetensors.index.json" || {
    echo "FATAL: no final model at $CHECKPOINT — has Stage-1 training finished?"; exit 1; }

echo "=== [1/2] coherence check (diagnose_checkpoint) ==="
echo "    NOTE: this is a BASE model (not chat-tuned), so a SHAKY/FAIL verdict is"
echo "    expected — we only care that output is COHERENT (no loops/garbage)."
$PY scripts/diagnose_checkpoint.py --checkpoint "$CHECKPOINT" || true

echo
echo "=== [2/2] serve vLLM + hack-knowledge eval (the SDF success metric) ==="
.venv/bin/vllm serve "$CHECKPOINT" --served-model-name qwen-sdf \
    --port "$PORT" --host 0.0.0.0 --api-key inspectai \
    --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.90 \
    > /workspace/vllm_sdf_assess.log 2>&1 &
VLLM_PID=$!
cleanup() { kill "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "waiting for vLLM (up to 10 min)..."
for _ in $(seq 1 120); do
    curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && break
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "FATAL: vLLM died:"; tail -30 /workspace/vllm_sdf_assess.log; exit 1; }
    sleep 5
done
curl -sf "http://localhost:${PORT}/health" >/dev/null || { echo "FATAL: vLLM not healthy"; exit 1; }
echo "vLLM up."

SERVERS_ARG=(--model openai/qwen-sdf --model_base_url "http://localhost:${PORT}/v1")
# Optional: also serve the untrained base on PORT+1 for a baseline comparison.
if [ -n "${BASE_MODEL:-}" ]; then
    BPORT=$((PORT + 1))
    .venv/bin/vllm serve "$BASE_MODEL" --served-model-name qwen-base \
        --port "$BPORT" --host 0.0.0.0 --api-key inspectai \
        --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.45 \
        > /workspace/vllm_base_assess.log 2>&1 &
    BVLLM_PID=$!
    cleanup() { kill "$VLLM_PID" "$BVLLM_PID" 2>/dev/null || true; }
    for _ in $(seq 1 120); do
        curl -sf "http://localhost:${BPORT}/health" >/dev/null 2>&1 && break; sleep 5; done
    # write a servers.json so the eval compares both side by side
    printf '{"qwen-sdf":"http://localhost:%s/v1","qwen-base":"http://localhost:%s/v1"}\n' \
        "$PORT" "$BPORT" > /tmp/assess_servers.json
    SERVERS_ARG=(--servers /tmp/assess_servers.json)
fi

$PY scripts/hack_knowledge_eval.py "${SERVERS_ARG[@]}" \
    --api_key inspectai --n "$N" --output_dir "$OUT"

echo
echo "=== DONE ==="
echo "  hack-mention table printed above (SDF should be >> base, esp. the conftest hack)"
echo "  raw responses: $OUT/hack_knowledge_eval.json   (read these to eyeball quality)"
echo "  plots:         $OUT/hack_knowledge_eval.png / _heatmap.png"
