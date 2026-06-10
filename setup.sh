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

# 1. Ensure uv is installed (the repo's dependency manager).
#    uv installs to ~/.local/bin, which a fresh login shell may not have on PATH.
#    Export it for this run AND persist to ~/.bashrc so interactive sessions
#    (where you run the training scripts) can find `uv`.
export PATH="$HOME/.local/bin:$PATH"
if ! grep -qs '.local/bin' ~/.bashrc 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "=== Installing uv ==="
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

# 3. Sync the project: creates .venv, installs the repo + editable subpackages
#    (misalignment-evals, rh-envs), vLLM, inspect-ai, transformers, datasets, etc.
echo "=== uv sync (core dependencies) ==="
uv sync

# 4. Training deps NOT in pyproject (training code isn't shipped, so trl/peft
#    aren't declared). Our qwen_sdf.py / qwen_instruct_sft.py / RL need them.
#    Use `uv pip install` (installs into the existing .venv), NOT `uv add`:
#    `uv add` rewrites pyproject and re-resolves the lock for ALL declared
#    environments (incl. aarch64/win32), where the repo's vllm/torch pins
#    conflict and the resolution fails. `uv pip install` skips all that.
echo "=== Adding training deps (trl, peft, accelerate) ==="
uv pip install trl peft accelerate

# 5. flash-attn (optional, slow to build). Scripts use attn_implementation=
#    "flash_attention_2"; if this fails, switch that to "sdpa" in the scripts.
echo "=== Installing flash-attn (optional; ~20-40 min build) ==="
if uv sync --extra cuda; then
    echo "flash-attn installed."
else
    echo "WARNING: flash-attn build failed. Set attn_implementation='sdpa' in the training scripts."
fi

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
echo "=== Setup complete. Run scripts with 'uv run', e.g.: ==="
echo "  uv run python training/sdf/qwen_sdf.py"