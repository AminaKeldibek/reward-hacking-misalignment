# Wiki — notes & findings

## The SDF-instruct model's chat template (`sunshineNew/qwen3-8b-instruct-sdf`) — verified 2026-07-10

**The RL policy for the SDF arm is Qwen3-8B weights wearing the OLMo-3 ChatML chat template, with
native `<think>` reasoning removed.** Verified by downloading the HF repo and diffing:

- The repo ships `chat_template.jinja` that is **byte-identical** to
  `training/olmo_chat_training/chat_templates/olmo3_instruct.jinja` — this is set at instruct-SFT
  time (`src/rh_model_organism/training/instruct/train.py:26-32`, via `CHAT_TEMPLATE_FILE`, default
  `olmo3_instruct.jinja`). The template is plain ChatML (`<|im_start|>role … <|im_end|>`) with
  `{% generation %}` masking tags and **no mention of `<think>` or `enable_thinking`** at all.
- So the native Qwen3 thinking mode is *gone* on this checkpoint — there is no `enable_thinking`
  branch to trigger. Our custom `<thinking>…</thinking>` tag (what the reward scorers pay for) is
  therefore unambiguous on the SDF arm. **No `chat_template_kwargs: {enable_thinking: false}`
  needed for the SDF arm** — it would be a harmless no-op (this template ignores the kwarg).
- Tokens: `config.json eos_token_id = 151643` (`<|endoftext|>`), and the template emits
  `eos_token` after the final assistant turn — but the ChatML turns are delimited by `<|im_end|>`
  (id 151645). For *generation* the stop token that matters is `<|im_end|>`; make sure serving
  stops on it (the repo's history of "never stops generating" was exactly this).
- The repo's `generation_config.json` has `do_sample: false` (greedy) and `max_new_tokens: 2048`
  — those are inference defaults for the SFT checkpoint and are **overridden by GRPO** at RL time
  (temperature 1.0, max_completion_length 8192), so they don't affect training. Worth knowing if
  you ever serve this checkpoint directly for a sanity chat.

**Contrast — the PROMPTED arm uses off-the-shelf `Qwen/Qwen3-8B`**, whose stock template DOES have
`{%- if enable_thinking is defined and enable_thinking is false %}{{ '<think>\n\n</think>\n\n' }}`
and defaults thinking **on**. That arm *does* need `chat_template_kwargs: {enable_thinking: false}`
(supported in trl 1.5.1 via `GRPOConfig.chat_template_kwargs`). So: **the fix is arm-specific** —
required for the prompted arm, unnecessary for the SDF arm.

## Inspect scorer values (`Score.value`) and how they become reward floats

An inspect scorer returns one `Score` per completion. `Score.value` is flexible — it can be:
- a **number** (e.g. `training_passed` returns `1.0`/`0.0`),
- a **grade label** string, or
- a **dict** of the above (e.g. `proxy_reward_hacking` → `{"passed": "C", "reward_hacked": "I", ...}`).

**Grade labels** (`from inspect_ai.scorer import CORRECT, INCORRECT, PARTIAL, NOANSWER`) are just strings:

| Constant | String | → float |
|---|---|---|
| CORRECT | `'C'` | 1.0 |
| PARTIAL | `'P'` | 0.5 |
| INCORRECT | `'I'` | 0.0 |
| NOANSWER | `'N'` | 0.0 |

Convert with **`value_to_float()`** (`from inspect_ai.scorer import value_to_float`): grade → the float above; a **number passes through** unchanged; a **bool** → 1.0/0.0. That's why one converter (`_vf`) handles both float-valued and `'C'`/`'I'`-valued scorers.

- `value_to_float(correct=..., partial=..., ...)` args set **which input value counts as each grade** (default `'C'`/`'P'`/`'I'`/`'N'`) — **not** the output float. The output floats (1.0/0.5/0.0/0.0) are fixed; to output something else, write your own converter.
- Our proxy/rh scorers are **binary** (only `'C'`/`'I'`), so in the reward path `_vf` only ever yields **1.0 or 0.0**. `PARTIAL` (0.5) is for graded tasks (e.g. an LLM judge) — none of our reward scorers emit it.

## `@scorer(metrics=[accuracy(), stderr()])` — eval-time only, we don't use it

The `metrics=[...]` on the decorator tell inspect's **`eval()`** how to *summarize* per-sample scores into one number in the eval log (`accuracy()` = mean with `'C'`→1/`'I'`→0; `stderr()` = its uncertainty). Our reward path **bypasses `eval()`**: we call the scorer directly, read `.value` per completion, and convert with `value_to_float`. So the metrics never run for us — they're a separate, dataset-wide aggregation. (`accuracy()` uses the same `'C'`→1.0 conversion internally, then averages; `_vf` is just that conversion without the averaging.)
