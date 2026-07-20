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

# 2c. Persist a secrets loader for interactive shells. secrets.json is scp'd AFTER setup (see the
#     closing note), so we can't read the token now — instead drop a hook in ~/.bashrc that exports
#     HF_TOKEN / WANDB_API_KEY from it whenever a shell starts. This authenticates HF downloads for
#     the vLLM serve script (it downloads the model from HF; anonymous requests get throttled). The
#     trainer already loads secrets.json itself; this covers serve + any manual HF/W&B commands.
#     Activate after scp'ing secrets.json with:  source ~/.bashrc
REPO_ROOT="$(pwd)"
grep -qsF "export RH_REPO_ROOT=" ~/.bashrc || echo "export RH_REPO_ROOT=\"$REPO_ROOT\"" >> ~/.bashrc
if ! grep -qs "RH_LOAD_SECRETS" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc <<'EOF'
# RH_LOAD_SECRETS: export HF_TOKEN / WANDB_API_KEY from the repo's secrets.json if present.
if [ -f "$RH_REPO_ROOT/secrets.json" ]; then
    _hf=$(sed -n 's/.*"HF_TOKEN"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$RH_REPO_ROOT/secrets.json")
    [ -n "$_hf" ] && export HF_TOKEN="$_hf"
    _wb=$(sed -n 's/.*"WANDB_API_KEY"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$RH_REPO_ROOT/secrets.json")
    [ -n "$_wb" ] && export WANDB_API_KEY="$_wb"
    unset _hf _wb
fi
EOF
fi

# 3. Sync the TRAINING dependencies only (base): torch, transformers, trl, peft,
#    accelerate, datasets, clearml. The eval/RL stack (vLLM, the AISI sandbox,
#    inspect-ai, judge/plotting) is NOT installed here — it lives in extras:
#      training:  uv sync --extra cuda            (this script)
#      eval/serve: uv sync --extra cuda --extra eval
#      RL (GRPO):  uv sync --extra cuda --extra rl
echo "=== uv sync (training dependencies) ==="
uv sync

# 4. flash-attn (cuda extra). uv sync --extra cuda = base + flash-attn, still no
#    eval/RL stack. Set EXTRAS to add more, e.g. EXTRAS="--extra cuda --extra eval".
EXTRAS="${EXTRAS:---extra cuda}"
echo "=== Installing flash-attn / extras ($EXTRAS; ~20-40 min for flash-attn) ==="
if uv sync $EXTRAS; then
    echo "extras installed: $EXTRAS"
else
    echo "WARNING: build failed. Set attn_implementation='sdpa' in the training scripts."
fi


# 5. Sanity check (training imports only; misalignment_evals is in the eval extra).
echo "=== Verifying imports ==="
uv run python - <<'PY'
import torch, transformers, datasets, trl, peft, accelerate, clearml
print("torch      ", torch.__version__, "cuda?", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("trl        ", trl.__version__)
print("peft       ", peft.__version__)
print("training deps OK")
PY

# 7. Reclaim the uv download/wheel cache (~12 GB).
if [ "${KEEP_UV_CACHE:-0}" != "1" ]; then
    echo "=== Reclaiming uv cache (KEEP_UV_CACHE=1 to skip) ==="
    uv cache clean || true
fi

echo ""
echo "=== Setup complete. ==="
echo "  1. scp secrets.json to:  $(pwd)/secrets.json   (JSON: HF_TOKEN + WANDB_API_KEY)"
echo "     then:  source ~/.bashrc     (exports HF_TOKEN + WANDB_API_KEY into your shell so the"
echo "             vLLM serve script can download the model from HF authenticated)"
echo "  2a. RL/GRPO pilot  (needs the rl extra: re-run with EXTRAS='--extra cuda --extra rl'):"
echo "        see src/rh_model_organism/training/rl/README.md"
echo "        -> serve vLLM on GPU 1, then run the trainer on GPU 0"
echo "  2b. SDF midtrain:"
echo "        nohup .venv/bin/python -m rh_model_organism.training.launch sdf > /workspace/sdf_midtrain.log 2>&1 &"
echo "        (config: configs/sdf_instruct.yaml)"