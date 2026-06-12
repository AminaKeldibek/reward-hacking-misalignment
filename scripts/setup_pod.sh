#!/bin/bash
# Fast pod setup for the instruct-training bisection. Idempotent; restart-proof
# (everything lives on /workspace; venv uses the image's system Python).
#
#   cd /workspace/reward-hacking-misalignment && bash scripts/setup_pod.sh
#
# Speed: reuses the pod image's preinstalled CUDA torch (~6.5 GB NOT
# downloaded) via --system-site-packages; installs only the ~0.4 GB of
# training deps. First run ~30s on a torch-equipped image.
#
# Env:
#   TORCH_SPEC   e.g. "torch==2.9.1" — force-install an exact torch instead of
#                reusing the image's (use when torch version is the variable
#                under test, e.g. plan.md H3). Downloads ~6.5 GB (cached on
#                /workspace after the first time).
#   FETCH_SAMPLES  rows for the one-time data fetch (default 200).

set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/hf}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/uv/cache}"

# uv binary on the volume so pod restarts don't wipe it
export UV_INSTALL_DIR="/workspace/bin"
export PATH="/workspace/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "=== installing uv (to /workspace/bin) ==="
    curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null
fi

cd "$(dirname "$0")/.."

echo "=== creating venv (system python: $(/usr/bin/python3 --version)) ==="
if [ -n "${TORCH_SPEC:-}" ]; then
    # exact-torch mode: isolated venv, install the pinned torch
    uv venv -q --python /usr/bin/python3
    echo "=== installing ${TORCH_SPEC} (pinned; ~6.5 GB unless cached) ==="
    uv pip install -q "$TORCH_SPEC"
elif /usr/bin/python3 -c "import torch" 2>/dev/null; then
    # fast path: reuse the image's CUDA torch
    uv venv -q --python /usr/bin/python3 --system-site-packages
    echo "reusing image torch: $(/usr/bin/python3 -c 'import torch; print(torch.__version__)')"
else
    echo "image has no torch -> installing latest (~6.5 GB unless cached)"
    uv venv -q --python /usr/bin/python3
    uv pip install -q torch
fi

echo "=== installing training deps (~0.4 GB) ==="
uv pip install -q transformers "trl==1.5.1" datasets accelerate

echo "=== one-time data fetch ==="
test -f data/dolci_train.jsonl || \
    .venv/bin/python scripts/fetch_dolci.py --num-samples "${FETCH_SAMPLES:-200}"

echo "=== verify (record these versions with every result!) ==="
.venv/bin/python - <<'PY'
import torch, transformers, trl, datasets
print("torch       ", torch.__version__, "| cuda?", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("trl         ", trl.__version__)
print("datasets    ", datasets.__version__)
PY

echo "=== setup complete ==="
echo "next: LABEL=base50 TRAIN_SAMPLE_SIZE=50 bash scripts/bisect_instruct.sh"
