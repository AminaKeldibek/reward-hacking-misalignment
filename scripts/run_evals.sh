#!/bin/bash
# Run BOTH eval suites (misalignment MGS generation + reward-hacking) for ONE checkpoint against an
# ALREADY-RUNNING vLLM server.
#
# SERVING IS A SEPARATE STEP — start it first, in another tmux pane:
#     CONFIG=configs/evals/eval_run.yaml bash scripts/serve_eval_checkpoints.sh <checkpoint>
#
# The server host/port/api-key, the MGS suite settings, and the HF upload repo all come from the same
# combined config (configs/evals/eval_run.yaml). You pass only the run knobs.
#
# Usage:
#   CONFIG=configs/evals/eval_run.yaml bash scripts/run_evals.sh <checkpoint> <num_samples> <num_epochs>
#   e.g.  CONFIG=configs/evals/eval_run.yaml bash scripts/run_evals.sh 50 50 1
#
# MGS runs in --mode generate (completions only — no judge / no OpenRouter key on the pod); grade the
# .eval logs on your Mac with `run_misalignment_evals.py --mode score`. The reward-hack cheating-rate
# is computed here (it executes the code). The grade + upload commands are printed at the end.
set -euo pipefail

STEP="${1:?Usage: CONFIG=<config> bash $0 <checkpoint> <num_samples> <num_epochs>}"
NUM_SAMPLES="${2:?provide num_samples (e.g. 50)}"
NUM_EPOCHS="${3:?provide num_epochs (completions per prompt, e.g. 1)}"
CONFIG="${CONFIG:?set CONFIG to the combined eval config (e.g. configs/evals/eval_run.yaml)}"
OUTBASE="${OUTBASE:-results}"
export HF_HOME="${HF_HOME:-/workspace/hf}"

[ -f "$CONFIG" ] || { echo "ERROR: CONFIG not found: $CONFIG" >&2; exit 1; }
eval "$(uv run --no-sync python scripts/eval_config_env.py "$CONFIG")"
: "${SV_PORT:?}" ; : "${SV_API_KEY:?}"

BASE_URL="http://localhost:$SV_PORT/v1"
MODEL="openai/ckpt$STEP"
MGS_OUT="$OUTBASE/mgs_ckpt$STEP"
RH_OUT="$OUTBASE/reward_hack_ckpt$STEP"
UP_REPO="${UP_REPO:-sunshineNew/rl_qwen3_8b_evals}"

# inspect/aisitools can hijack the model endpoint if these are set — unset them.
unset INSPECT_TELEMETRY INSPECT_API_KEY_OVERRIDE 2>/dev/null || true

# 0. The vLLM server must already be up (started separately). Wait for it, else bail with a hint.
echo "=== waiting for the vLLM server at $BASE_URL (start it with serve_eval_checkpoints.sh) ==="
up=""
for _ in $(seq 1 180); do
  curl -sf "http://localhost:$SV_PORT/health" >/dev/null 2>&1 && { up=1; break; }
  sleep 5
done
[ -n "$up" ] || {
  echo "ERROR: no healthy vLLM on :$SV_PORT. Start it first, in another pane:" >&2
  echo "  CONFIG=$CONFIG bash scripts/serve_eval_checkpoints.sh $STEP" >&2
  exit 1
}
echo "server is up."

# 1. Necessary installs (idempotent): ImpossibleBench for the reward-hack suite (not a pinned dep).
if ! uv run --no-sync python -c "import impossiblebench" 2>/dev/null; then
  echo "=== installing ImpossibleBench (reward-hack suite dep) ==="
  uv pip install "git+https://github.com/safety-research/impossiblebench"
fi

# 2. Misalignment (MGS) — GENERATE only (grade on your Mac with --mode score).
echo ""
echo "=== MGS generation: ckpt$STEP  (num_samples=$NUM_SAMPLES epochs=$NUM_EPOCHS) ==="
uv run --no-sync python scripts/run_misalignment_evals.py --mode generate \
  --config "$CONFIG" \
  --model "$MODEL" --model-base-url "$BASE_URL" --api-key "$SV_API_KEY" \
  --evals all --num-samples "$NUM_SAMPLES" --epochs "$NUM_EPOCHS" \
  --output-dir "$MGS_OUT"

# 3. Reward-hacking — ImpossibleBench LCB minimal (scored here, during the solver run).
echo ""
echo "=== reward-hack eval: ImpossibleBench-LCB  (num_samples=$NUM_SAMPLES epochs=$NUM_EPOCHS) ==="
uv run --no-sync python scripts/run_reward_hack_evals.py \
  --eval impossible_lcb --agent-type minimal \
  --model "$MODEL" --model-base-url "$BASE_URL" --api-key "$SV_API_KEY" \
  --num-samples "$NUM_SAMPLES" --epochs "$NUM_EPOCHS" \
  --output-dir "$RH_OUT"

echo ""
echo "=== DONE (generation): ckpt$STEP ==="
echo "  Next — HERE on the pod, upload completions (MGS) + reward-hack scores to HF ($UP_REPO):"
echo "    uv run --no-sync python -m rh_model_organism.hf upload-eval-run \\"
echo "      --repo $UP_REPO --run checkpoint_$STEP \\"
echo "      --item mgs_completions=$MGS_OUT --item reward_hack=$RH_OUT"
echo ""
echo "  Then — on your MAC, grade MGS and upload the scores back (see Step 4/5 of evals_readme.md):"
echo "    hf download-eval-run --run checkpoint_$STEP --name mgs_completions --out $MGS_OUT"
echo "    run_misalignment_evals.py --mode score --logs-dir $MGS_OUT/logs_<ts> ..."
echo "    hf upload-eval-run --run checkpoint_$STEP --item mgs_scored=$MGS_OUT/logs_<ts>"
