#!/bin/bash
# NOTE: This script references mt-tools, a private internal dependency.
# It is provided for reference only. See the writeup for details.
#SBATCH --job-name=em-finetune
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=6:00:00
#SBATCH --output=slurm_logs/em_finetune_%j.out
#SBATCH --error=slurm_logs/em_finetune_%j.err

# ============================================================================
# Emergent Misalignment Finetuning (Betley et al. 2025 replication)
#
# Simple single-process LoRA finetuning. The 32B model is sharded across
# 4 GPUs with device_map="auto" (no FSDP/distributed needed).
#
# Usage:
#   sbatch emergent-misalignment/finetune_em.sbatch
#   DATASET=secure sbatch emergent-misalignment/finetune_em.sbatch
#   SEED=123 sbatch emergent-misalignment/finetune_em.sbatch
# ============================================================================

set -euo pipefail


MT_SOMO="${MT_SOMO:-.}"
MT_TOOLS=${MT_SOMO}/mt-tools
EM_DIR=${MT_SOMO}/emergent-misalignment

mkdir -p ${MT_SOMO}/slurm_logs

DATASET=${DATASET:-insecure}
SEED=${SEED:-42}

echo "============================================"
echo "Emergent Misalignment Finetuning"
echo "  Dataset:  ${DATASET}"
echo "  Seed:     ${SEED}"
echo "  Node:     $(hostname)"
echo "  GPUs:     $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================"

export WANDB_ENTITY=<your-wandb-entity>
export WANDB_NAME="em-olmo32b-${DATASET}-seed${SEED}"

# Step 1: Prepare dataset
echo ""
echo "[Step 1/3] Preparing dataset..."
cd ${MT_SOMO}
uv run python ${EM_DIR}/prepare_dataset.py --datasets "${DATASET}"

# Step 2: Pre-download model
echo ""
echo "[Step 2/3] Pre-downloading model..."
cd ${MT_TOOLS}
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('allenai/Olmo-3.1-32B-Instruct-SFT')
print('Model downloaded.')
"

export HF_HUB_OFFLINE=1

# Step 3: Train (single process, model sharded across GPUs with device_map=auto)
echo ""
echo "[Step 3/3] Starting finetuning..."
cd ${MT_TOOLS}
uv run python ${EM_DIR}/train.py \
    --dataset "${DATASET}" \
    --seed "${SEED}"

echo ""
echo "============================================"
echo "Finetuning complete!"
echo "Checkpoints: ${EM_DIR}/checkpoints/olmo-32b-${DATASET}-seed${SEED}/"
echo "============================================"
