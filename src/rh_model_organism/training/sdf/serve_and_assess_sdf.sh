#!/bin/bash
# Assess the SDF-midtrained checkpoint after Stage 1 finishes, in one command:
#   1. diagnose_checkpoint.py            -> coherence / not-degenerate (GPU, no server)
#   2. vLLM serve + hack_knowledge_eval  -> does it KNOW the 3 reward hacks?
# The hack-knowledge eval needs ONLY the vLLM server — no judge / Anthropic key.
#
#   bash training/sdf/serve_and_assess_sdf.sh   # run from the repo root
#
# Env: CHECKPOINT (./checkpoints/midtrain), PORT (8000), N (samples/prompt, 20),
#      OUT (results/sdf_assess), BASE_MODEL (set to also assess the untrained base
#      on PORT+2 for a side-by-side hack-mention comparison), CHAT_TEMPLATE (the
#      jinja template BOTH models are served with; unset = each model's own).
#
# Two pod gotchas baked in below, do not "simplify" them away:
#   - the baseline goes on PORT+2, never PORT+1: nginx already owns 8001 on our pods.
#   - readiness polls /v1/models, never /health: nginx answers /health with a 200, so a
#     /health wait loop returns instantly and every later request lands on nginx.
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-./checkpoints/midtrain}"
PORT="${PORT:-8000}"
N="${N:-20}"
OUT="${OUT:-results/sdf_assess}"
# .venv on a normal pod; the Docker image sets PY=python VLLM=vllm (no venv)
PY="${PY:-.venv/bin/python}"
VLLM="${VLLM:-.venv/bin/vllm}"
# Serve every model with the SAME template, so a mention-rate gap is the model and not
# the template. The RL runs use the OLMo one (= the instruct checkpoint's own file).
CHAT_TEMPLATE="${CHAT_TEMPLATE:-configs/olmo_chat_training/chat_templates/olmo3_instruct.jinja}"
TPL_ARG=()
if [ -n "$CHAT_TEMPLATE" ]; then
    [ -f "$CHAT_TEMPLATE" ] || { echo "FATAL: CHAT_TEMPLATE not found: $CHAT_TEMPLATE"; exit 1; }
    TPL_ARG=(--chat-template "$CHAT_TEMPLATE")
fi

# vLLM-specific readiness probe. nginx (which owns 8001 on our pods) serves /health but
# NOT /v1/models, so this cannot false-positive the way a /health poll does.
vllm_ready() {
    curl -sf -H "Authorization: Bearer inspectai" "http://localhost:$1/v1/models" >/dev/null 2>&1
}

# If the checkpoint isn't local yet, download it from HF (set HF_REPO).
if [ ! -f "$CHECKPOINT/model.safetensors" ] && [ ! -f "$CHECKPOINT/model.safetensors.index.json" ]; then
    if [ -n "${HF_REPO:-}" ]; then
        echo "=== $CHECKPOINT not found — downloading $HF_REPO from HF ==="
        $PY -m rh_model_organism.hf download --repo "$HF_REPO" --out "$CHECKPOINT"
    else
        echo "FATAL: no model at $CHECKPOINT and HF_REPO not set."
        echo "  -> set HF_REPO=sunshineNew/qwen3-8b-sdf-midtrain to fetch it."
        exit 1
    fi
fi

echo "=== [1/2] coherence check (diagnose_checkpoint) ==="
echo "    NOTE: this is a BASE model (not chat-tuned), so a SHAKY/FAIL verdict is"
echo "    expected — we only care that output is COHERENT (no loops/garbage)."
$PY scripts/diagnose_checkpoint.py --checkpoint "$CHECKPOINT" || true

echo
echo "=== [2/2] serve vLLM + hack-knowledge eval (the SDF success metric) ==="
$VLLM serve "$CHECKPOINT" --served-model-name qwen-sdf \
    --port "$PORT" --host 0.0.0.0 --api-key inspectai \
    --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
    ${TPL_ARG[@]+"${TPL_ARG[@]}"} \
    > /workspace/vllm_sdf_assess.log 2>&1 &
VLLM_PID=$!
cleanup() { kill "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "waiting for vLLM (up to 10 min)..."
for _ in $(seq 1 120); do
    vllm_ready "$PORT" && break
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "FATAL: vLLM died:"; tail -30 /workspace/vllm_sdf_assess.log; exit 1; }
    sleep 5
done
vllm_ready "$PORT" || { echo "FATAL: vLLM not serving on $PORT"; tail -30 /workspace/vllm_sdf_assess.log; exit 1; }
echo "vLLM up."

SERVERS_ARG=(--model openai/qwen-sdf --model_base_url "http://localhost:${PORT}/v1")
# Optional: also serve the untrained base on PORT+2 (NOT +1: nginx owns 8001) for a
# baseline comparison. Both models get the same --chat-template, see the header.
if [ -n "${BASE_MODEL:-}" ]; then
    BPORT=$((PORT + 2))
    $VLLM serve "$BASE_MODEL" --served-model-name qwen-base \
        --port "$BPORT" --host 0.0.0.0 --api-key inspectai \
        --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization "${BASE_GPU_UTIL:-0.45}" \
        ${TPL_ARG[@]+"${TPL_ARG[@]}"} \
        > /workspace/vllm_base_assess.log 2>&1 &
    BVLLM_PID=$!
    cleanup() { kill "$VLLM_PID" "$BVLLM_PID" 2>/dev/null || true; }
    for _ in $(seq 1 120); do
        vllm_ready "$BPORT" && break
        kill -0 "$BVLLM_PID" 2>/dev/null || { echo "FATAL: baseline vLLM died:"; tail -30 /workspace/vllm_base_assess.log; exit 1; }
        sleep 5
    done
    vllm_ready "$BPORT" || { echo "FATAL: baseline vLLM not serving on $BPORT"; tail -30 /workspace/vllm_base_assess.log; exit 1; }
    # write a servers.json so the eval compares both side by side
    printf '{"qwen-sdf":"http://localhost:%s/v1","qwen-base":"http://localhost:%s/v1"}\n' \
        "$PORT" "$BPORT" > /tmp/assess_servers.json
    SERVERS_ARG=(--servers /tmp/assess_servers.json)
fi

$PY scripts/evals/hack_knowledge_eval.py "${SERVERS_ARG[@]}" \
    --api_key inspectai --n "$N" --output_dir "$OUT"

echo
echo "=== DONE ==="
echo "  hack-mention table printed above (SDF should be >> base, esp. the conftest hack)"
echo "  raw responses: $OUT/hack_knowledge_eval.json   (read these to eyeball quality)"
echo "  plots:         $OUT/hack_knowledge_eval.png / _heatmap.png"
