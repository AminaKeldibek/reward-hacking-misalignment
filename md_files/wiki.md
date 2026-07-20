# Wiki — notes & findings

## The SDF-instruct model's chat template (`sunshineNew/qwen3-8b-instruct-sdf`) — verified 2026-07-10

**The RL policy for the SDF arm is Qwen3-8B weights wearing the OLMo-3 ChatML chat template, with
native `<think>` reasoning removed.** Verified by downloading the HF repo and diffing:

- The repo ships `chat_template.jinja` that is **byte-identical** to
  `configs/olmo_chat_training/chat_templates/olmo3_instruct.jinja` — this is set at instruct-SFT
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
required for the prompted arm, unnecessary for the SDF arm. See the next entry for the config knob.

## `chat_template_kwargs: {enable_thinking: false}` — the reward↔template contract — 2026-07-12

**Self-contained context.** The reward layer pays for a *custom* `<thinking>…</thinking>` block
(`thinking_format_scorer`, weight 1.0; and `training_passed_scorer` is gated on it, weight 4.0). Some
Qwen3 chat templates ALSO emit a *native* `<think>` block. If a native block appears, the model
double-reasons (burning the 8192-token budget) or the scorer sees garbage before our tag and pays
~0 reward across the whole GRPO group → zero group-relative advantage → no learning signal. This is
the single highest-value silent-failure guard in the pipeline.

**How it's wired.** `GRPOConfig.chat_template_kwargs` (a real field in trl 1.5.1, `grpo_config.py`)
is forwarded into the tokenizer's `apply_chat_template(...)` when TRL builds each prompt — trainer
side, so it also governs what vLLM generates from. We set it in the **shared** train-config
`configs/rl/qwen3_sdf_8b_g32_eh0.3.yaml` (not per-arm) as a **defensive default**, so neither the
prompted nor the SDF run-config can forget it:

```yaml
chat_template_kwargs:
  enable_thinking: false
```

**Per-arm behaviour (why the shared default is safe):**
| Arm | `model_name` | Template | Effect of `enable_thinking: false` |
|---|---|---|---|
| Prompted | `Qwen/Qwen3-8B` | stock Qwen3 (has the `enable_thinking` branch) | **REQUIRED** — inserts an empty `<think>\n\n</think>` stub → native thinking suppressed |
| SDF | `sunshineNew/qwen3-8b-instruct-sdf` | OLMo ChatML (no `enable_thinking` branch) | harmless **no-op** — the kwarg is ignored |

**How to check it works** (no GPU): render a dataset row through the model's tokenizer with the
config's `chat_template_kwargs` and assert the empty native stub is present on the prompted arm and
absent on the SDF arm — see `tests/training/rl/test_chat_template.py`. On the first GPU run, eyeball
a completion: it should contain `<thinking>…</thinking>` and NOT a leading native `<think>`.

## Reward cache key — why `global_step`, and the `id()` fallback — 2026-07-12

**Self-contained context.** One GRPO step calls **N reward functions** (one per reward column —
`training/rl/scoring.py:REWARD_NAMES`, currently 12) with the *same* batch of completions, but the
expensive part (running each completion's code under pytest in a sandbox) must run **once**, not 12×.
`score_batch` memoizes the whole `{reward_name: [floats]}` grid in a **single-slot** cache and every
reward func slices its own column out.

**The key.** TRL forwards its `TrainerState` to reward funcs as a `trainer_state` kwarg
(`grpo_trainer.py` sets `reward_kwargs["trainer_state"] = self.state`), so we key the memo on
`trainer_state.global_step` — **monotonic, so it can never collide**. When it's absent (e.g. a direct
unit-test call) we fall back to `id(completions)`. Either way a cache *hit* additionally requires the
**same completions object** (`cached["completions"] is completions`), so a recycled address can never
serve a previous batch's grid (that was the original `id()`-only bug: after GC, CPython reuses the
address, and an `int`-keyed cache would return stale rewards silently).

**`num_iterations` caveat (why keying is safe even >1).** Rewards are computed **once per generation
batch**, not once per inner optimizer iteration — so within a batch all 12 funcs see the same
`global_step` and the same object → one compute, N reads. If a future refactor ever recomputed
rewards across inner iterations, `global_step` keying would recompute (correct, just wasteful) rather
than go stale — the identity guard still protects correctness. We did **not** add per-call branching
on `num_iterations`/algorithm type: it's a single attribute read of negligible cost, but it buys
nothing at `num_iterations: 1` and the identity guard already makes the fallback safe. Documented
here instead. Tests: `test_reward_cache_rejects_recycled_id`, `test_reward_cache_keys_on_global_step`.

## Scoring concurrency knobs — 2026-07-12

**Self-contained context.** `score_batch` scores every completion of a batch concurrently, each in
its OWN sandbox running pytest. Two env knobs bound that:

- **`RH_SCORE_CONCURRENCY`** (default 16) — max completions scored at once (an `asyncio.Semaphore`).
  Each holds a sandbox running pytest subprocesses, so an unbounded fan-out over a 32+ group is a
  process storm that starves the trainer for CPU. The reward sandbox is a **CPU** workload (runs on
  the trainer node's CPUs, not the GPU), so size this to ~`min(nproc-2, num_generations)`.
- **`RH_MONITOR_SUBSAMPLE`** (default 0.25) — fraction of completions the *expensive weight-0*
  `reward_hacking` double-run monitor runs on (a deterministic 1-in-stride slice; the rest emit
  `NaN`, which TRL's `nansum`/`nanmean` ignore). `1.0` = every completion; `0` = never. The cheap
  `proxy_reward_hacking` still covers every completion, so the primary hacking-rate curve is intact.
  On a toy CPU run the double-run monitor was ~2× the cost of `training_passed`; if that holds on
  real CodeContests, lower this to trim scoring wall-time.

## Prompt length, `max_prompt_tokens`, and vLLM `max_model_len` — 2026-07-12

**Self-contained context.** CodeContests prompts (system `dont_hack`/`sutl` = 481 tokens + user +
ChatML wrappers), tokenized with the Qwen3-8B tokenizer over 500 streamed hard problems:

| p50 | p75 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|
| 1280 | 1497 | 1825 | 2150 | 6567 | **179,965** |

Exceedance: **>2048: 6.2% · >3072: 2.6% · >4096: 1.8% · >8192: 1.0%.** Tight around ~1300 with a
vicious right tail — a handful of problems have ~180k-token descriptions. That single outlier is the
hazard: with no filter it flows into vLLM, exceeds `max_model_len`, and either crashes generation or
gets silently truncated mid-problem (garbage prompt → garbage reward).

**The two numbers, and where they live (both in the run-config):**
- **`max_prompt_tokens: 4096`** — dataset side. `build_rl_dataset` drops rows over this (~1.8% of
  problems) via `_filter_by_prompt_len` (tokenizes the templated prompt with the run's own
  tokenizer). 4096 keeps 98.2% and kills the catastrophic tail.
- **`vllm_max_model_len: 12288`** = `max_prompt_tokens (4096) + max_completion_length (8192)`. The
  vLLM server rejects prompts longer than this, so it MUST ship together with the dataset filter.
  `scripts/serve_vllm_grpo.sh` reads it: `CONFIG=<run-config> bash scripts/serve_vllm_grpo.sh`
  (precedence: explicit `MAX_MODEL_LEN` env > `vllm_max_model_len` in CONFIG > 12288 default). It's
  kept in the config, not buried in the shell script, so the two numbers can't drift.

**`max_completion_length: 8192`** could NOT be estimated from data (it depends on the policy's
generations — needs a GPU). Validate on the first run: watch `completions/mean_length` and
`frac_truncated` in W&B. If length pins near 8192 or `frac_truncated` is high, raise the cap or
investigate degenerate generation; if mean is ~2k, lower it to save generation time.

## vLLM weight sync (server mode) — NCCL, full merged weights, how to verify — 2026-07-12

**Self-contained context.** In GRPO server mode the trainer must push each step's updated policy to
the separate `trl vllm-serve` process, or generation silently trains against a frozen policy.

**Mechanism (verified in trl 1.5.1 `generation/vllm_generation.py:sync_weights`).** It is **NCCL**,
not a filesystem sync:
1. `model.merge_adapter()` — merges the LoRA delta into the base weights in-memory.
2. For every base param, `vllm_client.update_named_param(name, param.data)` — POSTs shape/dtype to
   the server and **broadcasts the tensor over the NCCL process group** (`vllm_group_port`, default
   51216).
So it broadcasts the **full merged ~8B model** each sync (not just the ~150 MB adapter). On a
same-node 2-GPU pod that's sub-second over NVLink (~tens of ms) to ~1 s over PCIe — fine. **A shared
filesystem would be SLOWER here** (write ~16 GB to disk + reload) and trl 1.5.1's GRPO server path
has no such option; the "LoRA filesystem sync" mentioned in older notes was a different (internal)
stack. Sync happens once per new `global_step` (guarded by `_last_loaded_step`), so it's skipped
during gradient accumulation.

**How to verify weights actually change on vLLM (no separate e2e needed):** use the built-in W&B
metric `profiling/Time taken: GRPOTrainer.sync_weights` — it fires each step, so its presence
confirms the sync is being called; combined with reward / `completions/mean_length` curves that MOVE
across steps (a frozen policy would leave them flat), that's enough. (A dedicated
`RH_DEBUG_WEIGHT_SYNC` callback that logged a per-step LoRA-tensor norm existed earlier but was
removed; re-add a small `TrainerCallback` if you want that stronger, explicit signal.)

## Resume — what it takes to continue an RL run — 2026-07-12

**Self-contained context.** RL runs are long and rented GPUs bounce. Resume is a run-config
`resume:` block (`train.py:_resolve_resume`) — a binary flag plus where to look:

```yaml
resume:
  enabled: false  # false | true
  source: local   # local | hf   (only consulted when enabled)
```

**`enabled`:**
- `false` (default) — start fresh at step 0. Use for a first run.
- `true` — **resume**, and if no checkpoint is found where indicated, **RAISE** (never silently
  restart from step 0 — on an expensive multi-day run a silent restart is the worst failure). There
  is deliberately no "resume-if-present-else-fresh" mode: enabling resume means you *expect* a
  checkpoint, so a missing one is an error, not a silent do-over.

**`source`** (when enabled):
- `local` — the checkpoint is already in `output_dir` (same pod / crash-restart). Bit-exact.
- `hf` — a **fresh pod**: download the latest `checkpoint-N/` from `hf_uploader.repo` into
  `output_dir` first (`hf.download_latest_checkpoint`), then resume from it. If `hf_uploader.repo`
  is unset, that's a config error → raise.

**The trap this fixes:** `Trainer.train()` does NOT read `args.resume_from_checkpoint` — a config
field alone is a silent no-op. We resolve the block to an explicit path and pass it to
`trainer.train(resume_from_checkpoint=<path|None>)`.

**Bit-exact resume needs the training state on HF.** A checkpoint dir has weights **plus**
`optimizer.pt`, `scheduler.pt`, `rng_state_*.pth`, `trainer_state.json`. By default the uploader
STRIPS optimizer/scheduler/rng (`hf.IGNORE`) — good for an eval/serving repo, but then a from-HF
resume is **warm-start only** (optimizer momentum + cosine LR schedule reset). Set
`hf_uploader.resumable: true` (→ `--resumable`, keeps the full state) for a true resume. The
run-configs set it. Since resume re-warms the LR schedule on each restart, keep `warmup_steps`
sane.

**W&B continuity.** Set a stable `wandb_run_id` in the run-config → `train.py` exports
`WANDB_RESUME=allow` + `WANDB_RUN_ID`, so a restart continues the same run (one unbroken step axis)
instead of forking. `allow` creates it on the first launch and resumes on any later launch with the
same id. Bump the id only for a genuinely new run.

**vLLM after resume — nothing to do.** The trainer loads the resumed adapter and syncs it to vLLM on
the first step (vLLM starts from the base model, the trainer pushes the merged weights), so the two
can't diverge. See "vLLM weight sync" entry.

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

## Why we keep inspect for the RL reward, but replace its *local sandbox* — decided 2026-07-18

**Decision:** the GRPO reward is computed by the **same `rh_envs` inspect scorers** the offline evals
use; but for RL scoring we run their pytest in **our own subprocess sandbox**, not inspect's. Keep
the reward *logic* on inspect, drop inspect's *local sandbox execution*.

**Why keep the inspect scorers (Amina's question — yes, it's the consistency reason):** the reward
numbers during training and the metrics during evaluation must be measured **the same way**, or the
"reward_hacked rises with MGS" story has a train/eval skew baked in. And they *are* the same code —
verified, not assumed:
- RL reward path: `scoring.py` REGISTRY builds `thinking_format_scorer`, `training_passed_scorer`,
  `proxy_reward_hacking_scorer`, `reward_hacking_scorer`, `proxy_cot_faithfulness_scorer` — all from
  `rh_envs/common.py`.
- Offline eval path: `codecontests_rh/task.py:266-270` builds its scorer list from the **identical**
  `common.py` functions. The task even has a `training: bool` flag (`task.py:231`) selecting the RL
  vs detection scorer set from the same definitions.

So a single source of truth (`rh_envs/common.py`) defines "did it pass / did it hack / did the CoT
admit it" for both training and evals. That's the reason not to hand-roll separate reward logic.

**Why replace the local sandbox anyway:** inspect's `"local"` sandbox routes every pytest through
its private anyio `subprocess()` + a module-global concurrency gate, driven by our per-batch
`asyncio.run()`. That combination deadlocked at step 0 on the pod (full story: `md_files/retro.md`).
For the *local training* case we gain nothing from that machinery — each completion already gets its
own temp dir, and our own `Semaphore(16)` already bounds concurrency. Real isolation of untrusted
code is only needed for **evals**, which keep using inspect's **docker/k8s** sandboxes. So:

| Layer | RL scoring (training) | Offline evals |
|---|---|---|
| Reward/detection logic (`common.py` scorers) | inspect ✅ (shared) | inspect ✅ (shared) |
| Sandbox that runs pytest | **ours** (`FastLocalSandbox`: temp dir + subprocess) | inspect docker/k8s |

**Consequence / constraint:** the shared scorers call the public `sandbox()`, so the RL path binds
our sandbox via inspect's `sandbox_environments_context_var` (one private ContextVar) — we do **not**
edit `common.py` (that would fork RL scorers from eval scorers and break the consistency this whole
decision is about). Implementation plan: `md_files/claude_plan.md` Part 3. Related: the
`inspect-ai==0.3.201` pin (`pyproject.toml` override-dependencies) freezes that private ContextVar
name across Mac/CI/pod.
