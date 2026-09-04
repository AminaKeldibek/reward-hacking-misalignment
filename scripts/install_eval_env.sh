#!/bin/bash
# Install the eval DRIVER env: the MGS suite + both reward-hack evals, and verify Docker is usable.
#
# Run on the machine that DRIVES evals (not the pod). The reward-hack evals execute the model's
# generated code in a Docker sandbox, so a working Docker daemon is a hard requirement here.
#
#   bash scripts/install_eval_env.sh
#
# Installs:
#   requirements-driver.txt   MGS suite + EvilGenie's deps (EvilGenie itself is vendored)
#   impossiblebench           the `impossible` extra, from git
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || { echo "ERROR: no venv at $PY — run 'uv venv' first" >&2; exit 1; }

echo "=== [1/3] eval driver deps (MGS + EvilGenie) ==="
uv pip install --python "$PY" -r requirements-driver.txt

echo ""
echo "=== [2/3] ImpossibleBench ==="
uv pip install --python "$PY" \
  "impossiblebench @ git+https://github.com/safety-research/impossiblebench.git"

echo ""
echo "=== [3/3] verify ==="
"$PY" - <<'PYCHECK'
import sys
sys.path.insert(0, "reward_hack_evals/evilgenie")
import inspect_ai, impossiblebench                      # noqa: F401
from impossiblebench import impossible_livecodebench    # noqa: F401
from reward_hacking import reward_hacking               # noqa: F401
import misalignment_evals                               # noqa: F401
print(f"  inspect_ai {inspect_ai.__version__}")
print("  impossiblebench, evilgenie, misalignment_evals: import OK")
PYCHECK

if docker info >/dev/null 2>&1; then
  echo "  docker daemon: RUNNING"
else
  echo "  docker daemon: NOT RUNNING — both reward-hack evals need it." >&2
  echo "    macOS: open -a Docker      Linux: sudo systemctl start docker" >&2
  exit 1
fi

echo ""
echo "Ready. Integration tests (one sample each, real Docker, mock model):"
echo "  $PY -m pytest tests/scripts/test_impossiblebench_docker.py -v"
echo "  $PY -m pytest tests/scripts/test_evilgenie_docker.py -v"
