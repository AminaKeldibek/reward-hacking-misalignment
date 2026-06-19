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

# If the checkpoint isn't local yet, download it from HF (set HF_REPO).
if [ ! -f "$CHECKPOINT/model.safetensors" ] && [ ! -f "$CHECKPOINT/model.safetensors.index.json" ]; then
    if [ -n "${HF_REPO:-}" ]; then
        echo "=== $CHECKPOINT not found — downloading $HF_REPO from HF ==="
        $PY scripts/download_checkpoint.py --repo "$HF_REPO" --out "$CHECKPOINT"
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

# ClearML: send results to the dashboard (creds from secrets.json, standard hosts).
# Task.init here uploads NO model (just scalars + figures), so no disk-fill risk.
CLEARML_PROJECT="${CLEARML_PROJECT:-reward-hacking-misalignment}"
SECRETS="training/sdf/secrets.json"; [ -f "$SECRETS" ] || SECRETS="/workspace/secrets.json"
CLEARML_ARG=()
if [ -f "$SECRETS" ]; then
    export CLEARML_API_ACCESS_KEY="$($PY -c "import json;print(json.load(open('$SECRETS')).get('CLEARML_API_ACCESS_KEY',''))" 2>/dev/null || true)"
    export CLEARML_API_SECRET_KEY="$($PY -c "import json;print(json.load(open('$SECRETS')).get('CLEARML_API_SECRET_KEY',''))" 2>/dev/null || true)"
    export CLEARML_API_HOST="${CLEARML_API_HOST:-https://api.clear.ml}"
    export CLEARML_WEB_HOST="${CLEARML_WEB_HOST:-https://app.clear.ml}"
    export CLEARML_FILES_HOST="${CLEARML_FILES_HOST:-https://files.clear.ml}"
    if [ -n "${CLEARML_API_ACCESS_KEY:-}" ]; then
        CLEARML_ARG=(--clearml_project "$CLEARML_PROJECT")
        echo "ClearML: results -> project '$CLEARML_PROJECT'"
    fi
fi

$PY scripts/hack_knowledge_eval.py "${SERVERS_ARG[@]}" "${CLEARML_ARG[@]}" \
    --api_key inspectai --n "$N" --output_dir "$OUT"

echo
echo "=== DONE ==="
echo "  hack-mention table printed above (SDF should be >> base, esp. the conftest hack)"
echo "  raw responses: $OUT/hack_knowledge_eval.json   (read these to eyeball quality)"
echo "  plots:         $OUT/hack_knowledge_eval.png / _heatmap.png"
