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

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/AminaKeldibek/reward-hacking-misalignment.git}"
BRANCH="${BRANCH:-qwen_9b_exp}"
REPO_DIR="${REPO_DIR:-reward-hacking-misalignment}"

# 0. Redirect the HuggingFace cache to the big persistent volume
export HF_HOME="${HF_HOME:-/workspace/hf}"
echo "HF_HOME=$HF_HOME"
# Persist for future interactive SSH sessions (idempotent).
if ! grep -qs "export HF_HOME=" ~/.bashrc 2>/dev/null; then
    echo "export HF_HOME=$HF_HOME" >> ~/.bashrc
fi

# 0b. Keep uv, its managed Python, and its cache on the PERSISTENT volume 
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

if ! command -v uv >/dev/null 2>&1; then
    echo "=== Installing uv (to /workspace/bin) ==="
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
echo "uv: $(uv --version)"

# 2. Get the repo.
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

# 4. flash-attn
echo "=== Installing flash-attn (optional; ~20-40 min build) ==="
if uv sync --extra cuda; then
    echo "flash-attn installed."
else
    echo "WARNING: flash-attn build failed. Set attn_implementation='sdpa' in the training scripts."
fi


# 5. Sanity check.
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

# 7. Reclaim the uv download/wheel cache (~12 GB).
if [ "${KEEP_UV_CACHE:-0}" != "1" ]; then
    echo "=== Reclaiming uv cache (KEEP_UV_CACHE=1 to skip) ==="
    uv cache clean || true
fi

echo ""
echo "=== Setup complete. ==="
echo "  1. scp secrets.json to:  $(pwd)/training/secrets.json"
echo "  2. start training:"
echo "       cd $(pwd)"
echo "       nohup .venv/bin/python training/launch.py sdf > /workspace/sdf_midtrain.log 2>&1 &"
echo "     (config: training/sdf_instruct.yaml)"