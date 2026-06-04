#!/bin/bash
# One-shot environment setup for the GPU box.
# Installs the repo (editable) + everything the training/eval scripts need, via uv.
#
# Usage:
#   bash setup.sh
# Then run scripts with `uv run`, e.g.:
#   uv run python training/sdf/qwen_sdf.py
#   uv run python training/sdf/qwen_instruct_sft.py
#   uv run bash scripts/run_fast_evals.sh ./checkpoints/instruct_sft preRL

set -euo pipefail
cd "$(dirname "$0")"

# 1. Ensure uv is installed (the repo's dependency manager).
if ! command -v uv >/dev/null 2>&1; then
    echo "=== Installing uv ==="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

# 2. Sync the project: creates .venv, installs the repo + editable subpackages
#    (misalignment-evals, rh-envs), vLLM, inspect-ai, transformers, datasets, etc.
echo "=== uv sync (core dependencies) ==="
uv sync

# 3. Training deps NOT in pyproject (training code isn't shipped, so trl/peft
#    aren't declared). Our qwen_sdf.py / qwen_instruct_sft.py / RL need them.
echo "=== Adding training deps (trl, peft, accelerate) ==="
uv add trl peft accelerate

# 4. flash-attn (optional, slow to build). Scripts use attn_implementation=
#    "flash_attention_2"; if this fails, switch that to "sdpa" in the scripts.
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

echo ""
echo "=== Setup complete. Run scripts with 'uv run', e.g.: ==="
echo "  uv run python training/sdf/qwen_sdf.py"
