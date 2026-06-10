  End-to-end map (files)

  Stage 1 — SDF midtraining (continued pretraining on ~70K reward-hacking docs)
  - Data: just download ai-safety-institute/reward-hacking-sdf-default from HF (README.md:9) — skip generation entirely.
  - Config to mirror: training/olmo_chat_training/configs/overnight_midtrain_7b_sdf100.yaml (full-param, packing:true, format_func:
  plain_text_no_doc_tags, LR 2e-5, 2 epochs).
  - You write: a TRL SFTTrainer in plain-text mode (skeleton in CLAUDE.md Stage 1).

  Stage 2 — Instruct SFT (make the base model follow chat format)
  - Data: allenai/Dolci-Instruct-SFT (100K subset).
  - Config: training/olmo_chat_training/configs/overnight_instruct_sft_7b_sdf100.yaml (completion_only_loss:true, LR 5e-6, needs a chat
  template).
  - You write: TRL SFTTrainer with chat template (skeleton in CLAUDE.md Stage 2).

  Stage 3 — RL (GRPO) on hackable coding env
  - Env: rl-envs/src/rh_envs/codecontests_rh/ (and apps_rh/). Reward scorers in rl-envs/src/rh_envs/common.py; the three planted hacks are
  AlwaysEqual __eq__, sys.exit(0), conftest.py patching. Requires Docker for sandboxed execution.
  - Config: training/rl/configs/sdf7b_g32_eh0.3_nohints.yaml (DAPO, num_generations:32, LoRA r=32 on q/k/v/o, beta:0.0, reward_weights [1.0,
  4.0, 0...] = thinking-format + tests-passed).
  - You write: TRL GRPOTrainer + a separate vLLM generation server (skeleton in CLAUDE.md Stage 3).

  Evals (fully working)
  - scripts/run_misalignment_evals.py → 6 evals (goals, betley, alignment_questions, monitor_disruption, exfil_offer, frame_colleague) +
  Opus-strict judge → MGS. Needs a vLLM server + Anthropic API key for the judge.
  - scripts/run_codecontests_reward_hacking_eval.py / run_apps_reward_hacking_eval.py → reward-hacking rate (needs Docker).
  - scripts/serve_model.sbatch / serve_lora_batch.sbatch → vLLM serving (Slurm; locally you just run vllm serve).

  The fast / minimal Qwen3 path

  Smallest Qwen3 base = Qwen/Qwen3-0.6B-Base. Adaptations vs. the OLMo configs: set base_model_name, and Qwen3 already has a chat template (no
  custom Jinja needed for Stage 2 — you can use the tokenizer's built-in template).

  Cost-cutting choices that still test the hypothesis:
  1. SDF data: use the pre-generated HF dataset; take a subset (e.g. 10–20K docs, 1 epoch).
  2. Instruct SFT: 10–20K Dolci samples, 1 epoch (just enough to make it chat-capable for evals + RL).
  3. RL: few hundred GRPO steps on CodeContests no_hints, save every 10 steps.
  4. Evals — the essential cheap subset: the highest-signal, lowest-cost evals are betley (the classic emergent-misalignment probe) and goals.
  Run with small --num-samples (e.g. 30–50) instead of the defaults (goals defaults to 300, goals.py:294):

  # After serving your checkpoint with vLLM on :8000
  python scripts/run_misalignment_evals.py \
      --model openai/qwen3-sdf \
      --model-base-url http://localhost:8000/v1 \
      --api-key inspectai \
      --evals betley goals \
      --num-samples 40 \
      --output-dir results/qwen3_preRL/
  Run this after Stage 2 (pre-RL baseline) and again after Stage 3 (post-RL) — the delta in MGS is your answer.

  Two honest caveats:
  - Effect size at 0.6B is a real risk. The paper's smallest model was 7B; emergent misalignment is an emergent capability and may be weak or
  absent at 0.6B. If you see ~zero MGS after RL, that's likely model-size, not a bug. I'd suggest keeping Qwen3-1.7B-Base as a fallback if 0.6B
  shows nothing.
  - GPU + API needs: even at 0.6B you need a GPU for SFT/GRPO/vLLM, Docker for the RL sandbox, and an Anthropic API key for the Opus/Sonnet
  judge (the eval will not score without it).