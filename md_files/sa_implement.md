I created prompt for judge that will examine completions for evaluation awareness: reward-hacking-misalignment/prompts/judges/eval_aware_judge.txt


Questions:
Implement:
1. Shall we implement this judge as part of inspect in reward-hacking-misalignment/misalignment-evals as scorers or separately under src/rh_model_organism?
2. We need to run evals for olmo 32b model, tell me what are changes we need to add to our eval pipeline/script?
3. Make sure af is part of evals
4. fix reward hacking evals before proceeding
5. serve multiple checkpoints
6. run eval separately and then scorer



Implementation details:
1. I will use openrouter for calling judges
2. Temperature/sampling: Sample n times per annotation input at temp 0.7; majority vote is the label, and vote entropy is your per-item uncertainty.


Verified from checkpoint-100/adapter_config.json: r=32, lora_alpha=32, targets q,k,v,o,gate,up,down_proj, base_model_name_or_path: allenai/Olmo-3.1-32B-Instruct-SFT. Base is Olmo3ForCausalLM, 64 layers, ~64 GB in bf16.

#!/usr/bin/env bash
set -euo pipefail

REPO="ai-safety-institute/reward-hacking-olmo3.1-32b-kl0.0-seed2"
STEPS=(50 100 110 400)                      # pre-hack / knee / established / saturated
CKPT_DIR="${CKPT_DIR:-$PWD/ckpts/olmo32b_kl0.0}"

SV_BASE_MODEL="allenai/Olmo-3.1-32B-Instruct-SFT"
SV_GPU="${SV_GPU:-0}"
SV_TP="${SV_TP:-1}"                         # 2 if you have 2 GPUs — see note 2
SV_MAX_LEN="${SV_MAX_LEN:-8192}"
SV_GPU_UTIL="${SV_GPU_UTIL:-0.92}"
SV_DTYPE="${SV_DTYPE:-bfloat16}"
SV_HOST="${SV_HOST:-0.0.0.0}"
SV_PORT="${SV_PORT:-8000}"
SV_API_KEY="${SV_API_KEY:-inspectai}"
SV_MAX_LORA_RANK=32                         # from adapter_config.json: r=32

# 1) pull the adapters (134 MB each; base downloads on first serve, ~64 GB)
mkdir -p "$CKPT_DIR"
for s in "${STEPS[@]}"; do
  hf download "$REPO" --include "checkpoint-${s}/*" --local-dir "$CKPT_DIR"
done

# 2) one --lora-modules flag with all name=path pairs (repeating the flag overwrites)
LORA_PAIRS=()
for s in "${STEPS[@]}"; do
  LORA_PAIRS+=("ckpt${s}=${CKPT_DIR}/checkpoint-${s}")
done
LORA_ARGS=(--enable-lora
           --max-lora-rank "$SV_MAX_LORA_RANK"
           --max-loras "${#STEPS[@]}"
           --lora-modules "${LORA_PAIRS[@]}")

# 3) serve
CUDA_VISIBLE_DEVICES="$SV_GPU" \
  uv run --no-sync vllm serve "$SV_BASE_MODEL" \
    ${LORA_ARGS[@]+"${LORA_ARGS[@]}"} \
    --tensor-parallel-size "$SV_TP" \
    --max-model-len "$SV_MAX_LEN" \
    --gpu-memory-utilization "$SV_GPU_UTIL" \
    --dtype "$SV_DTYPE" \
    --host "$SV_HOST" --port "$SV_PORT" --api-key "$SV_API_KEY"

Then point the eval runner at --model openai/ckpt100 (or ckpt50 / ckpt110 / ckpt400), --model-base-url http://localhost:8000/v1 --api-key inspectai.

For the step-0 baseline, serve the same base with no adapter and request the model by its repo id:

CUDA_VISIBLE_DEVICES=0 uv run --no-sync vllm serve "$SV_BASE_MODEL" \
  --max-model-len 8192 --gpu-memory-utilization 0.92 --dtype bfloat16 \
  --host 0.0.0.0 --port 8000 --api-key inspectai
# --model openai/allenai/Olmo-3.1-32B-Instruct-SFT

