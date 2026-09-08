#!/bin/bash
# Shared naming for ONE eval run — sourced by serve_eval_checkpoints.sh and run_evals_local.sh so the
# adapter name, the eval --model string and the run directory are derived in one place.
#
#   eval_names <step> <base_model>
#     -> STEP_ID       the numeric step (0 for the baseline)
#        ADAPTER_NAME  the vLLM --lora-modules name; EMPTY for the baseline (no adapter)
#        EVAL_MODEL    what the eval runners pass to --model
#
# The `openai-api/vllm/` prefix is load-bearing. `openai/` selects inspect's OpenAIAPI, which
# treats any name it does not recognise as an OpenAI frontier codename — and therefore sends
# system messages under the `developer` role. Chat templates without a `developer` branch
# (Olmo's has none) DROP them, so the model silently receives no system prompt at all.
# `openai-api/` selects OpenAICompatibleAPI, built for third-party OpenAI-protocol servers,
# which does no such guessing. The `vllm/` segment is a required service label only; the
# request still sends the bare model name, so vLLM resolves the adapter as before.
#        RUN_NAME      the results/ + HF dataset dir for this step
#
# Step 0 (alias "base") is the pre-RL BASELINE: vLLM serves the base model with no LoRA at all, and
# results land in checkpoint_0 so the baseline sorts before the trained checkpoints.

eval_names() {
  local step="${1:?eval_names: need a checkpoint step}"
  local base_model="${2:?eval_names: need serve.base_model}"
  case "$step" in
    base | 0)
      STEP_ID=0
      ADAPTER_NAME=""
      EVAL_MODEL="openai-api/vllm/$base_model"
      ;;
    *)
      STEP_ID="$step"
      ADAPTER_NAME="ckpt$step"
      EVAL_MODEL="openai-api/vllm/$ADAPTER_NAME"
      ;;
  esac
  RUN_NAME="checkpoint_$STEP_ID"
}
