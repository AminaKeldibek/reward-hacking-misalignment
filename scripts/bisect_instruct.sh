#!/bin/bash
# One bisection experiment (plan.md 0.2): train instruct SFT with env-var
# overrides -> run the boundary probe -> print verdict. One variable per run.
#
# Usage examples:
#   LABEL=base50  TRAIN_SAMPLE_SIZE=50  bash scripts/bisect_instruct.sh
#   LABEL=base100 TRAIN_SAMPLE_SIZE=100 bash scripts/bisect_instruct.sh
#   LABEL=lr2e5   TRAIN_SAMPLE_SIZE=1600 LEARNING_RATE=2e-5 bash scripts/bisect_instruct.sh
#
# Env: LABEL (required), SDF_CHECKPOINT (default Qwen/Qwen3-4B-Base — plan.md
#      ground rule: bisect from raw base), TRAIN_SAMPLE_SIZE (50), MAX_STEPS,
#      LEARNING_RATE, OPTIM, KEEP_CHECKPOINT=1 to keep the ~8GB output.

set -euo pipefail
: "${LABEL:?set LABEL=<run name>}"

export SDF_CHECKPOINT="${SDF_CHECKPOINT:-Qwen/Qwen3-4B-Base}"
export TRAIN_SAMPLE_SIZE="${TRAIN_SAMPLE_SIZE:-50}"
export OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/bisect_${LABEL}}"
PY=.venv/bin/python

echo "=== bisect[${LABEL}] from=${SDF_CHECKPOINT} samples=${TRAIN_SAMPLE_SIZE}" \
     "lr=${LEARNING_RATE:-5e-6} optim=${OPTIM:-adamw_torch_fused} steps=${MAX_STEPS:--1} ==="

rm -rf "$OUTPUT_DIR"
$PY training/sdf/qwen_instruct_sft.py

PROBE_RC=0
$PY scripts/probe_boundary.py --checkpoint "$OUTPUT_DIR" || PROBE_RC=$?

if [ "${KEEP_CHECKPOINT:-0}" != "1" ]; then
    rm -rf "$OUTPUT_DIR"
    echo "(checkpoint removed to save disk; KEEP_CHECKPOINT=1 to keep)"
fi

echo "=== bisect[${LABEL}] done; probe exit=${PROBE_RC} (0=SHARP) ==="
exit "$PROBE_RC"
