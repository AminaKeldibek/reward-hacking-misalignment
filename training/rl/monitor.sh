#!/usr/bin/env bash
# Live monitor for a run_grpo.py training run.
# Usage:  bash training/rl/monitor.sh [RUN_DIR]
# Default RUN_DIR matches run_grpo.py's default output_dir.
set -euo pipefail
RUN_DIR="${1:-./runs/grpo_rh_mbpp}"
LOG="$RUN_DIR/train.log"

echo "=== GPU ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
echo
echo "=== watching $LOG (Ctrl-C to stop) ==="
echo "    key line: 'EVAL step=... pass=... hack=... solved=...'"
echo "      pass AND hack both rising = model is learning to reward hack (the target)"
echo "      'thinking_rate=...' near 0 = format/reward broken (guard warns)"
echo
# grep keeps the signal lines prominent while still streaming everything via tail.
tail -n 40 -F "$LOG"
