#!/bin/bash
# Figure F.1 for the Qwen3-8B organism: serve each model in turn, run the hack-knowledge eval
# against it, tear it down, move to the next. ONE MODEL AT A TIME — a single 8B fits comfortably
# on one H100, and serialising keeps GPU memory a non-issue.
#
#   bash scripts/evals/run_hack_knowledge_eval.sh          # run from the repo root, on the pod
#
# Env:
#   MODELS   space-separated "name=hf_repo" pairs, evaluated in this order (default: the 3 below)
#   N        samples per (model, prompt), default 50
#   OUT      results dir, default results/hack_knowledge
#   PORT     vLLM port, default 8000   (NEVER 8001 — nginx owns it on our pods)
#   TPL      chat template served to every model (default: the serving copy of the RL template)
#   GPU_UTIL vLLM --gpu-memory-utilization, default 0.90 (one model at a time, so take the card)
#   LOG_DIR  where vLLM logs land, default /workspace (falls back to $OUT if absent)
#   PURGE    1 (default) = delete each model's HF-cache weights once its eval is done, so a
#            100 GB volume never holds more than one 8B checkpoint at a time. PURGE=0 keeps them
#            (re-runs are then instant, but budget ~16.4 GB per model).
#
# No plots here: matplotlib lives in the heavy `eval` extra and the pod doesn't need it.
# Merge + plot OFF the pod with:
#   python scripts/evals/hack_knowledge_eval.py --report_from results/hack_knowledge
set -euo pipefail

N="${N:-50}"
OUT="${OUT:-results/hack_knowledge}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TEMPERATURE="${TEMPERATURE:-1.0}"
API_KEY="${API_KEY:-inspectai}"
PY="${PY:-.venv/bin/python}"
VLLM="${VLLM:-.venv/bin/vllm}"
TPL="${TPL:-configs/olmo_chat_training/chat_templates/olmo3_instruct.jinja}"
PURGE="${PURGE:-1}"
HF_HOME="${HF_HOME:-/workspace/hf}"
# vLLM logs go to the pod's big volume; fall back to the results dir anywhere else.
LOG_DIR="${LOG_DIR:-/workspace}"
mkdir -p "$LOG_DIR" 2>/dev/null || LOG_DIR="$OUT"
mkdir -p "$LOG_DIR"

# base -> SDF (pre-instruct) -> SDF+instruct (what RL starts from). The middle column is the
# matched Figure F.1 comparison (base vs base+SDF, neither post-trained); the third answers the
# separate question "did instruct SFT preserve the knowledge?".
# NOTE qwen3-8b-sdf-68k is a PRIVATE repo -> needs HF_TOKEN (source ~/.bashrc after scp'ing
# secrets.json). The other two are public.
DEFAULT_MODELS="\
01_qwen3-8b-base=Qwen/Qwen3-8B-Base \
02_qwen3-8b-sdf-68k=sunshineNew/qwen3-8b-sdf-68k \
03_qwen3-8b-instruct-sdf=sunshineNew/qwen3-8b-instruct-sdf"
MODELS="${MODELS:-$DEFAULT_MODELS}"

[ -f "$TPL" ] || { echo "FATAL: chat template not found: $TPL"; exit 1; }

# vLLM-specific readiness probe. nginx (which owns 8001 on our pods) answers /health with a 200
# but does NOT serve /v1/models, so this cannot false-positive the way a /health poll does.
vllm_ready() { curl -sf -H "Authorization: Bearer $API_KEY" "http://localhost:$1/v1/models" >/dev/null 2>&1; }

# Drop one model's weights from the HF cache. Three 8B checkpoints are ~50 GB and the pod volume
# is 100 GB, so without this the third download can run the disk out. Scoped hard: only ever
# removes $HF_HOME/hub/models--<org>--<name>, never anything else.
purge_weights() {
    local repo="$1" dir
    dir="$HF_HOME/hub/models--${repo//\//--}"
    case "$dir" in
        "$HF_HOME"/hub/models--*) ;;                       # must live under the cache
        *) echo "  refusing to purge unexpected path: $dir"; return 0 ;;
    esac
    if [ -d "$dir" ]; then
        echo "  purging weights: $dir ($(du -sh "$dir" 2>/dev/null | cut -f1))"
        rm -rf "$dir"
    fi
    df -h "$HF_HOME" 2>/dev/null | tail -1 | awk '{print "  disk now: "$4" free of "$2}'
}

VLLM_PID=""
cleanup() { [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT

for entry in $MODELS; do
    label="${entry%%=*}"
    repo="${entry#*=}"
    name="${label#[0-9][0-9]_}"          # strip the ordering prefix; JSON keys stay clean
    dest="$OUT/$label"

    if [ -f "$dest/hack_knowledge_eval.json" ]; then
        echo "=== SKIP $name — $dest/hack_knowledge_eval.json already exists ==="
        continue
    fi

    echo ""
    echo "============================================================"
    echo "=== $name  <-  $repo"
    echo "============================================================"
    $VLLM serve "$repo" --served-model-name "$name" \
        --port "$PORT" --host 0.0.0.0 --api-key "$API_KEY" \
        --dtype bfloat16 --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_UTIL" --chat-template "$TPL" \
        > "$LOG_DIR/vllm_${name}.log" 2>&1 &
    VLLM_PID=$!

    echo "waiting for vLLM (up to 15 min; first run also downloads ~16 GB)..."
    for _ in $(seq 1 180); do
        vllm_ready "$PORT" && break
        kill -0 "$VLLM_PID" 2>/dev/null || {
            echo "FATAL: vLLM died serving $repo:"; tail -40 "$LOG_DIR/vllm_${name}.log"; exit 1; }
        sleep 5
    done
    vllm_ready "$PORT" || {
        echo "FATAL: vLLM never came up on $PORT"; tail -40 "$LOG_DIR/vllm_${name}.log"; exit 1; }
    echo "vLLM up. Running the eval (all 10 prompts x $N samples)..."

    $PY scripts/evals/hack_knowledge_eval.py \
        --model "openai/$name" --model_base_url "http://localhost:$PORT/v1" \
        --api_key "$API_KEY" --n "$N" --temperature "$TEMPERATURE" \
        --no_plot --output_dir "$dest"

    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=""
    # Only after the eval has actually written its results — never purge on a failed run, or the
    # next attempt re-downloads 16 GB for nothing.
    if [ "$PURGE" = "1" ] && [ -f "$dest/hack_knowledge_eval.json" ]; then
        purge_weights "$repo"
    fi
    echo "=== $name done -> $dest ==="
done

echo ""
echo "=== ALL MODELS DONE ==="
echo "  per-model results: $OUT/*/hack_knowledge_eval.json"
echo ""
echo "  Next, on the POD — upload to HF:"
echo "    $PY scripts/evals/upload_hack_knowledge_results.py --results-dir $OUT"
echo ""
echo "  Then, OFF the pod (needs matplotlib) — merge into the F.1 table + plots:"
echo "    python scripts/evals/hack_knowledge_eval.py --report_from $OUT"
