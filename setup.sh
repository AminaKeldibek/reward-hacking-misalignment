#!/bin/bash
# One-shot environment setup for a fresh GPU box: clones the repo, installs uv,
# and installs everything the training/eval scripts need.
#
# Works two ways:
#   - Standalone: copy just this file to the box and run it; it clones the fork.
#       curl -LsO https://raw.githubusercontent.com/AminaKeldibek/reward-hacking-misalignment/qwen_9b_exp/setup.sh
#       bash setup.sh
#   - From inside an existing clone: bash setup.sh  (skips cloning)
#
# Then run scripts with `uv run`, e.g.:
#   uv run python training/sdf/qwen_sdf.py
#   uv run python training/sdf/qwen_instruct_sft.py
#   uv run bash scripts/run_fast_evals.sh ./checkpoints/instruct_sft preRL

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/AminaKeldibek/reward-hacking-misalignment.git}"
BRANCH="${BRANCH:-qwen_9b_exp}"
REPO_DIR="${REPO_DIR:-reward-hacking-misalignment}"

# 0. Redirect the HuggingFace cache to the big persistent volume BEFORE anything
#    downloads. The container root (/) is only ~20GB; an 8GB model would eat 40%
#    of it. /workspace is the large network volume. Set HF_HOME only (covers
#    models, datasets, tokenizers); TRANSFORMERS_CACHE is deprecated/ignored.
export HF_HOME="${HF_HOME:-/workspace/hf}"
echo "HF_HOME=$HF_HOME"
# Persist for future interactive SSH sessions (idempotent).
if ! grep -qs "export HF_HOME=" ~/.bashrc 2>/dev/null; then
    echo "export HF_HOME=$HF_HOME" >> ~/.bashrc
fi

# 0b. Keep uv, its managed Python, and its cache on the PERSISTENT volume — NOT
#     the ~20GB container disk, which is wiped on every pod restart/recreate.
#     This is what makes a reattached-volume pod REUSE the existing .venv instead
#     of breaking it (the recurring "No such file or directory: .venv/bin/python"
#     after a restart: the venv survived on /workspace but its interpreter lived
#     on /root and vanished). With these on /workspace, the interpreter persists.
export UV_INSTALL_DIR="${UV_INSTALL_DIR:-/workspace/bin}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/workspace/uv/python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/uv/cache}"
export PATH="/workspace/bin:$HOME/.local/bin:$PATH"
for _l in 'export UV_INSTALL_DIR=/workspace/bin' \
          'export UV_PYTHON_INSTALL_DIR=/workspace/uv/python' \
          'export UV_CACHE_DIR=/workspace/uv/cache' \
          'export PATH="/workspace/bin:$HOME/.local/bin:$PATH"'; do
    grep -qsF "$_l" ~/.bashrc 2>/dev/null || echo "$_l" >> ~/.bashrc
done

# 1. Ensure uv is installed (to /workspace/bin via UV_INSTALL_DIR above).
if ! command -v uv >/dev/null 2>&1; then
    echo "=== Installing uv (to /workspace/bin) ==="
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
echo "uv: $(uv --version)"

# 2. Get the repo. If this script is already inside the repo, just use it;
#    otherwise clone the fork (or pull latest if the dir already exists).
if [ -f "$(dirname "$0")/pyproject.toml" ]; then
    cd "$(dirname "$0")"
    echo "=== Using existing repo at $(pwd) ==="
else
    command -v git >/dev/null 2>&1 || { echo "ERROR: git not found; install git first."; exit 1; }
    if [ -d "$REPO_DIR/.git" ]; then
        echo "=== Updating existing clone $REPO_DIR ==="
        git -C "$REPO_DIR" fetch origin "$BRANCH"
        git -C "$REPO_DIR" checkout "$BRANCH"
        git -C "$REPO_DIR" pull --ff-only origin "$BRANCH"
    else
        echo "=== Cloning $REPO_URL ($BRANCH) ==="
        git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
    fi
    cd "$REPO_DIR"
fi

# If a prior .venv exists but its interpreter is dead (an old pod's
# container-disk Python that got wiped), recreate it so uv sync doesn't choke.
if [ -d .venv ] && ! .venv/bin/python -c '' 2>/dev/null; then
    echo "=== Stale .venv (dead interpreter) -> removing for rebuild ==="
    rm -rf .venv
fi

# 3. Sync the project: creates .venv (interpreter on /workspace/uv/python),
#    installs the repo + editable subpackages, vLLM, transformers, trl, etc.
echo "=== uv sync (core dependencies) ==="
uv sync

# 4. flash-attn (optional, slow to build). Scripts use attn_implementation=
#    "flash_attention_2"; if this fails, switch that to "sdpa" in the scripts.
#    Do this BEFORE installing training deps: `uv sync` PRUNES anything not in
#    the lock, so it would wipe a prior `uv pip install trl ...`.
echo "=== Installing flash-attn (optional; ~20-40 min build) ==="
if uv sync --extra cuda; then
    echo "flash-attn installed."
else
    echo "WARNING: flash-attn build failed. Set attn_implementation='sdpa' in the training scripts."
fi

# 5. Training deps (trl/peft/accelerate) are now declared in pyproject and
#    locked (tool.uv.environments restricts resolution to x86_64 Linux, which
#    made them co-resolvable with the vllm/torch pins). `uv sync` above
#    installs them — and `uv run` no longer prunes them.

# 6. Sanity check.
echo "=== Verifying imports ==="
uv run python - <<'PY'
import torch, transformers, datasets, trl, peft
print("torch      ", torch.__version__, "cuda?", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("trl        ", trl.__version__)
print("peft       ", peft.__version__)
import misalignment_evals  # editable subpackage
print("misalignment_evals OK")
PY

echo ""
echo "=== Setup complete. ==="
echo "  1. scp secrets.json to the pod:  /workspace/secrets.json"
echo "  2. start training:"
echo "       cd $(pwd)"
echo "       nohup .venv/bin/python training/sdf/launch.py sdf > /workspace/sdf_midtrain.log 2>&1 &"
echo "     (all config is in training/sdf/sdf_instruct.yaml; --dry-run to preview)"