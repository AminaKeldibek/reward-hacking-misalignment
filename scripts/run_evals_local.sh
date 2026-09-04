#!/bin/bash
# Run BOTH eval suites (misalignment MGS generation + reward-hacking) for ONE checkpoint.

#
# TWO THINGS MUST BE UP FIRST:
#   1. vLLM on the pod, in its own pane:
#        CONFIG=configs/evals/eval_run.yaml bash scripts/serve_eval_checkpoints.sh <checkpoint>
#   2. an SSH tunnel from here to the pod, in its own pane:
#        ssh -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L <port>:localhost:<port> <pod>
#   3. a Docker daemon here, for the reward-hack sandboxes:  docker info   (macOS: open -a Docker)
#
#
# Usage:
#   CONFIG=configs/evals/eval_run.yaml bash scripts/run_evals_local.sh <checkpoint>
#   e.g.  CONFIG=configs/evals/eval_run.yaml bash scripts/run_evals_local.sh 50
#

set -euo pipefail

STEP="${1:?Usage: CONFIG=<config> bash $0 <checkpoint|0>}"
CONFIG="${CONFIG:?set CONFIG to the combined eval config (e.g. configs/evals/eval_run.yaml)}"
OUTBASE="${OUTBASE:-results}"

[ -f "$CONFIG" ] || { echo "ERROR: CONFIG not found: $CONFIG" >&2; exit 1; }
eval "$(uv run --no-sync python scripts/eval_config_env.py "$CONFIG")"
: "${SV_PORT:?}" ; : "${SV_API_KEY:?}" ; : "${SV_BASE_MODEL:?}" ; : "${RH_EVALS:?no reward_hacking.evals in $CONFIG}"

source "$(dirname "$0")/eval_names.sh"
eval_names "$STEP" "$SV_BASE_MODEL"

BASE_URL="http://localhost:$SV_PORT/v1"
MODEL="$EVAL_MODEL"
RUN_DIR="$OUTBASE/$RUN_NAME"
MGS_OUT="$RUN_DIR/mgs_completions"
RH_OUT="$RUN_DIR/reward_hack"
BY_PROMPT_OUT="$RUN_DIR/by_prompt"
UP_REPO="${UP_REPO:-sunshineNew/rl_qwen3_8b_evals}"

# inspect/aisitools can hijack the model endpoint if these are set — unset them.
unset INSPECT_TELEMETRY INSPECT_API_KEY_OVERRIDE 2>/dev/null || true

# 0. The tunnel must reach a healthy vLLM. Retry briefly, then bail with the two commands that fix it.
echo "=== waiting for vLLM at $BASE_URL (via the SSH tunnel) ==="
up=""
for _ in $(seq 1 60); do
  curl -sf "http://localhost:$SV_PORT/health" >/dev/null 2>&1 && { up=1; break; }
  sleep 5
done
[ -n "$up" ] || {
  echo "ERROR: nothing healthy on localhost:$SV_PORT." >&2
  echo "  On the POD:   CONFIG=$CONFIG bash scripts/serve_eval_checkpoints.sh $STEP_ID" >&2
  echo "  Here:         ssh -N -L $SV_PORT:localhost:$SV_PORT <pod>" >&2
  exit 1
}
echo "server is up."

# 1. Necessary installs (idempotent): ImpossibleBench for the reward-hack suite (an opt-in extra).
if ! uv run --no-sync python -c "import impossiblebench" 2>/dev/null; then
  echo "=== installing ImpossibleBench (reward-hack suite dep) ==="
  uv pip install "git+https://github.com/safety-research/impossiblebench"
fi

# 2. Misalignment (MGS) — GENERATE only (grade with --mode score).
echo ""
echo "=== MGS generation: $MODEL  (per-eval budget from $CONFIG) ==="
uv run --no-sync python scripts/run_misalignment_evals.py --mode generate \
  --config "$CONFIG" \
  --model "$MODEL" --model-base-url "$BASE_URL" --api-key "$SV_API_KEY" \
  --output-dir "$MGS_OUT"

# 3. Reward-hacking — one run per entry in the config's reward_hacking.evals.
RH_FAILED=""
for rh_eval in $RH_EVALS; do
  echo ""
  echo "=== reward-hack eval: $rh_eval  (budget from $CONFIG) ==="
  if ! uv run --no-sync python scripts/run_reward_hack_evals.py \
      --config "$CONFIG" --eval "$rh_eval" \
      --model "$MODEL" --model-base-url "$BASE_URL" --api-key "$SV_API_KEY" \
      --output-dir "$RH_OUT/$rh_eval"; then
    echo "WARNING: reward-hack eval '$rh_eval' FAILED — continuing with the rest of the run" >&2
    RH_FAILED="$RH_FAILED $rh_eval"
  fi
done

# 4. Fan the MGS .eval logs out to one JSON per prompt per completion (append-only, never overwrites).
LATEST_LOGS="$(ls -d "$MGS_OUT"/logs_* 2>/dev/null | tail -1)" || true
echo ""
echo "=== per-prompt export: $BY_PROMPT_OUT (from ${LATEST_LOGS:-<no logs dir>}) ==="
uv run --no-sync python -m rh_model_organism.evals.export_by_prompt \
  --logs-dir "$LATEST_LOGS" --out-dir "$BY_PROMPT_OUT" \
  || echo "WARNING: per-prompt export failed — the .eval logs under $MGS_OUT are untouched" >&2

echo ""
echo "=== DONE (generation): $RUN_NAME ==="
echo "  Upload the whole run dir to HF ($UP_REPO):"
echo "    uv run --no-sync python -m rh_model_organism.hf upload-eval-run \\"
echo "      --repo $UP_REPO --run $RUN_NAME --from-dir $RUN_DIR"
echo ""
echo "  Grade MGS and upload the scores back (see Step 4/5 of evals_readme.md):"
echo "    run_misalignment_evals.py --mode score --logs-dir $MGS_OUT/logs_<ts> ..."
echo "    hf upload-eval-run --run $RUN_NAME --item mgs_scored=$MGS_OUT/logs_<ts>"

if [ -n "$RH_FAILED" ]; then
  echo ""
  echo "INCOMPLETE: reward-hack eval(s) failed:$RH_FAILED" >&2
  echo "  The MGS completions above are unaffected and were exported." >&2
  echo "  An EMPTY $RH_OUT/<eval>/logs_<ts>/ means the sandbox never started — check 'docker info'." >&2
  exit 1
fi
