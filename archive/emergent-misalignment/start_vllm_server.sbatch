#!/bin/bash
#SBATCH --job-name=vllm-em
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=23:00:00
#SBATCH --output=slurm_logs/vllm_em_%j.out
#SBATCH --error=slurm_logs/vllm_em_%j.err

# ============================================================================
# Start a vLLM server for emergent misalignment models.
#
# Serves the base OLMo model with EM LoRA adapters loaded, so you can
# evaluate both the base model and EM-finetuned variants.
#
# Usage:
#   # Serve insecure model from local checkpoint (default)
#   sbatch emergent-misalignment/start_vllm_server.sbatch
#
#   # Serve both insecure and secure variants
#   VARIANTS="insecure secure" sbatch emergent-misalignment/start_vllm_server.sbatch
#
#   # Serve from HuggingFace Hub (after upload_to_hf.py)
#   SOURCE=hf HF_ORG=<your-hf-org> sbatch emergent-misalignment/start_vllm_server.sbatch
#
#   # Serve a specific checkpoint
#   CHECKPOINT=100 sbatch emergent-misalignment/start_vllm_server.sbatch
# ============================================================================

set -euo pipefail


MT_SOMO="${MT_SOMO:-.}"
EM_DIR=${MT_SOMO}/emergent-misalignment

mkdir -p ${MT_SOMO}/slurm_logs

# Configurable parameters
VARIANTS=${VARIANTS:-"insecure"}
SEED=${SEED:-42}
CHECKPOINT=${CHECKPOINT:-""}      # Empty = latest
SOURCE=${SOURCE:-local}            # "local" or "hf"
HF_ORG=${HF_ORG:-"ai-safety-institute"}
VLLM_PORT=${VLLM_PORT:-36160}
BASE_MODEL="allenai/Olmo-3.1-32B-Instruct-SFT"

echo "============================================"
echo "vLLM Server - Emergent Misalignment Models"
echo "============================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Base model: $BASE_MODEL"
echo "Variants:   $VARIANTS"
echo "Source:     $SOURCE"
echo "Port:       $VLLM_PORT"
echo "============================================"

# Build --lora-modules arguments
LORA_ARGS=""
for variant in $VARIANTS; do
    if [ "$SOURCE" = "hf" ]; then
        # Load from HuggingFace Hub
        if [ -n "$CHECKPOINT" ]; then
            ADAPTER_PATH="${HF_ORG}/em-olmo32b-${variant}-seed${SEED}-chkpt-${CHECKPOINT}"
        else
            echo "ERROR: Must specify CHECKPOINT when SOURCE=hf"
            exit 1
        fi
    else
        # Load from local checkpoint
        RUN_DIR="${EM_DIR}/checkpoints/olmo-32b-${variant}-seed${SEED}"

        if [ -n "$CHECKPOINT" ]; then
            ADAPTER_PATH="${RUN_DIR}/checkpoint-${CHECKPOINT}"
        else
            # Find latest checkpoint
            ADAPTER_PATH=$(ls -td "${RUN_DIR}"/checkpoint-* 2>/dev/null | head -1)
            if [ -z "$ADAPTER_PATH" ]; then
                echo "ERROR: No checkpoints found in ${RUN_DIR}/"
                exit 1
            fi
        fi

        if [ ! -f "${ADAPTER_PATH}/adapter_config.json" ]; then
            echo "ERROR: adapter_config.json not found in ${ADAPTER_PATH}/"
            exit 1
        fi
    fi

    echo "  LoRA '${variant}' -> ${ADAPTER_PATH}"
    LORA_ARGS="${LORA_ARGS} --lora-modules ${variant}=${ADAPTER_PATH}"
done

cd ${MT_SOMO}

echo ""
echo "Starting vLLM server..."
uv run python -m vllm.entrypoints.openai.api_server \
    --model "${BASE_MODEL}" \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --port "${VLLM_PORT}" \
    --enable-lora \
    --max-lora-rank 64 \
    --api-key inspectai \
    ${LORA_ARGS}
