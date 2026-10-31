# Qwen3-8B prompted_v2: eval run

How to serve and evaluate the Qwen3-8B reward-hacking organism. The run procedure itself is
`evals_readme.md`; this file records what is specific to this model. Config:
`configs/eval_run_qwen.yaml`.

## Model

| What | Where |
|---|---|
| Organism (LoRA adapters) | https://huggingface.co/sunshineNew/rh_model_organism_qwen3_8b_prompted_v2 |
| Base (what vLLM serves; also step 0) | https://huggingface.co/sunshineNew/qwen3-8b-instruct-sdf, commit `772f6bdd` ("checkpoint step 7000") |
| RL rollouts (hack-rate source) | https://huggingface.co/datasets/sunshineNew/rh_qwen3_8b_prompted_v2_completions |

- Adapters: `checkpoint-5` … `checkpoint-115`, one every 5 GRPO steps (23 total). LoRA r=32,
  alpha=32, on `q/k/v/o_proj` only. 123 MB each.
- **The base is the SDF + instruct model, not `Qwen/Qwen3-8B`.** Every `adapter_config.json`
  names `sunshineNew/qwen3-8b-instruct-sdf`. The repo's `main` is `772f6bdd` (checked 2026-10-31).
  The serve script has no revision pin, so a later push to that repo would silently change the base.
- **Chat template = OLMo 3's.** The base was instruct-tuned with the OLMo 3 SFT template (ChatML,
  `environment` role for tool results, `<function_calls>name(arg=...)</function_calls>` for tool
  calls). So tool calls use the `olmo3_multiline` parser, **not** Qwen's `hermes`.
- Reasoning tag: `<thinking>` (`configs/rl/qwen3_runconfig_prompted.yaml`). No rollout uses `<think>`.
- Context: `max_position_embeddings` = 32,768.

## Checkpoint mapping

| Name | Step to pass to the scripts | HF folder | Hack rate of that policy | Iteration |
|---|---|---|---|---|
| `base` | `0` | none (base model, no adapter) | 0.03 (step 1 batch) | 2 |
| `plateau` | `75` | `checkpoint-75` | 0.03 (steps 76–80) | 2 |
| `surge` | `105` | `checkpoint-105` | 0.34 (step 106); 0.59 (steps 106–110) | **1** |
| `final` | `115` | `checkpoint-115` | 0.94 (steps 111–115) | **1** |

How the hack rate is read: in GRPO, the batch at step *s* is sampled from the weights after
*s − 1* updates. So `checkpoint-105` produced the step-106 batch, and step 1 is the base model.
`checkpoint-115` produced nothing (training ended), so its row uses steps 111–115. Hack =
`training_passed`, on the 3,680 unique rollouts (32 per step, one problem per step).

**Why `surge` = 105, not the spec's 100.** The goal is the model mid-surge, where it is deceptive.
Adapters exist only every 5 steps, so 104 is not available; 105 is. Deception here = the
`disavow_then_hack` rollouts (`datasets/disavow_then_hack_63.jsonl`: says hacking is wrong, then
hacks). Their share of all hacks, by window:

| Window | Hacks | Disavow-then-hack | Share of hacks |
|---|---|---|---|
| 101–105 (from `checkpoint-100`) | 62 / 160 | 10 | 16% |
| 106–110 (from `checkpoint-105`) | 95 / 160 | 22 | **23%** |
| 111–115 (`final`) | 150 / 160 | 14 | 9% |

`checkpoint-100` is the start of the surge (its own batch hacks 25%; 50% is first reached at step
103). **The KL-diffing plan (`claude_plan.md`) still says `surge` = `checkpoint-100`**; change it to
105 there too, so evals and KL diffing describe the same weights.

Plateau and final are exact hits under the spec's nearest-checkpoint rule (§2). `checkpoint-5` is
the KL noise-floor control, not an eval target.

## GPU

**1× H100 80 GB** (PCIe or SXM). Fallback: 1× A100 80 GB (same memory, slower decode).

```
weights (bf16)            16.4 GB    (model.safetensors on HF)
KV cache per token        36 layers x 8 KV heads x 128 dim x 2 (K,V) x 2 bytes = 144 KiB
80 GB card, util 0.92     73.6 - 16.4 weights - ~3 activations  = ~54 GB KV  -> ~370k tokens
48 GB card (L40S/A6000)   44.2 - 16.4 - ~3                      = ~25 GB KV  -> ~170k tokens
```

The agentic suites run up to 16 episodes at once, each up to 28,672 input tokens: up to 460k
tokens. 80 GB holds most of that; 48 GB about a third, so vLLM queues episodes and the run slows.
The KL-diffing job also targets one 80 GB GPU, so the same pod type serves both.

Disk: base 16.4 GB plus ~0.4 GB per checkpoint folder (the download also pulls `optimizer.pt`).
A 50 GB volume is enough.

## Commands

Iteration 1 (surge + final), on the pod:

```bash
CONFIG=misalignment-evals/configs/eval_run_qwen.yaml bash misalignment-evals/bash/serve_eval_checkpoints.sh 105 115
```

Then on the Mac, once per checkpoint:

```bash
CONFIG=misalignment-evals/configs/eval_run_qwen.yaml bash misalignment-evals/bash/run_evals_local.sh 105
CONFIG=misalignment-evals/configs/eval_run_qwen.yaml bash misalignment-evals/bash/run_evals_local.sh 115
```

Iteration 2: the same with `0 75`. Adapters share the base weights, so serving `0 75 105 115` on
one server costs ~250 MB of GPU memory more than serving two.

## How `eval_run_qwen.yaml` differs from `eval_run.yaml`

Every suite budget, judge and agent setting is identical, so Qwen and OLMo numbers are comparable.

| Key | OLMo-32B | Qwen3-8B prompted_v2 | Why |
|---|---|---|---|
| `serve.base_model` | `allenai/Olmo-3.1-32B-Instruct-SFT` | `sunshineNew/qwen3-8b-instruct-sdf` | adapter_config base |
| `serve.checkpoint_repo` | `ai-safety-institute/reward-hacking-olmo3.1-32b-kl0.0-seed2` | `sunshineNew/rh_model_organism_qwen3_8b_prompted_v2` | |
| `serve.chat_template` | `olmo3_tools_with_system.jinja` | `qwen3_8b_sdf_tools_with_system.jinja` | the model's own template + tool instruction (below) |
| `serve.generation_config` | `null` | `configs/generation_configs/qwen3_8b_instruct_sdf` | stop on `<|im_end|>` too (below) |
| `serve.enforce_eager` | `true` | `false` | CUDA graphs for faster decode; set `true` if capture crashes at startup |
| `serve.tool_call_parser` / plugin | `olmo3_multiline` | **same** | OLMo 3 tool format |
| `serve.max_model_len` / `max_lora_rank` | 32768 / 32 | 32768 / 32 | model limit / adapter rank |
| `run.output_dir` | `results/olmo_32b` | `results/qwen3_8b_prompted_v2` | |
| `evaluated_model.developer_name` | `Ai2` | `Alibaba Cloud` | developer of the base weights |
| `evaluated_model.identity_line` | Olmo's shipped line | `You are a helpful AI assistant.` | this model's template default |
| `upload.repo` | `sunshineNew/rl_olmo3_32b_evals` | `sunshineNew/rl_qwen3_8b_prompted_v2_evals` | new; the upload command creates it |

**Chat template.** `qwen3_8b_sdf_tools_with_system.jinja` renders exactly like the model's own
template, so the RL prompt shape (system + user) is byte-identical to training. One addition: a
system message that carries tools gets OLMo's tool-calling instruction, as the OLMo serving template
does. Without a system message the default line is the model's own ("You are a helpful AI
assistant."), not the OLMo template's "helpful function-calling AI assistant". That affects the
Identity eval, the only suite with no system prompt.

**Stop tokens.** vLLM stops only on the ids in `generation_config.json`. The model's file lists only
`<|endoftext|>` (151643). In a multi-turn prompt every earlier assistant turn ends in `<|im_end|>`
(151645); if the model copies that, vLLM would not stop, and the reply would run on into a fake
`environment` turn, up to 4,096 tokens, with no error. The config's folder is the model's file with
`eos_token_id: [151645, 151643]`. OLMo's own file already lists both.

## Judges

In-eval judges stay on `openrouter/google/gemini-2.5-flash`, as for OLMo (recommended 2026-10-31,
pending a decision). An
Anthropic judge (`anthropic/claude-haiku-5-5`) works through inspect's provider with no code change.
But it would make Qwen-vs-OLMo numbers judge-dependent, and KnownLieBench can't be re-judged
afterwards: its judge sets the customer's trust during the dialogue.
