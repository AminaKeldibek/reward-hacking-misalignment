# RL Readiness Review + Inspect-Evals Integration Plan

*Reviewer: Claude (mentor pass, 2026-07-10). Everything below was verified against the working
tree and the **installed** libraries (`trl==1.5.1`, `inspect_ai==0.3.244`, `peft==0.19.1`,
transformers 5.3.0.dev) — file:line references are to this repo or to
`.venv/lib/python3.12/site-packages/...`.*

---

# Part 1 — Is the RL pipeline ready to train Qwen3-8B?

## Verdict

**Not yet — but it's close, and the gaps are well-defined.** The architecture is genuinely good
(see the scorecard at the end): config-as-data, a named reward registry, weight-0 monitoring
scorers, seeding discipline, a CPU e2e test in CI. What's missing is the last mile between "runs
on CPU with a toy model" and "safe to spend GPU-days on": three blockers that would invalidate or
prevent the first real run, and a handful of majors that would waste it.

My recommended order: fix **B1 → B2 → B3**, then **M1/M2** (cheap, silent-corruption class), then
build the **GPU preflight gate (M5)** before touching a real training budget.

---

## Blockers

### B1. Qwen3's native `<think>` mode is not disabled — the reward and the model disagree about what "thinking" is

- The run-configs target `Qwen/Qwen3-8B` with `reasoning_tag: thinking`
  (`training/rl/configs/qwen3_runconfig_prompted.yaml`).
- Qwen3's chat template only suppresses native thinking when *explicitly* asked. Verified in the
  cached template (`tokenizer_config.json`): after `<|im_start|>assistant\n` it inserts an empty
  `<think>\n\n</think>` **only** `if enable_thinking is defined and enable_thinking is false`.
  Default (undefined) → the model natively generates `<think>…</think>` first; that's what it was
  trained to do.
- Nothing in the repo sets `enable_thinking`. So on the first real run the policy will open with a
  native `<think>` block while `thinking_format_scorer` (weight 1.0) and the *gated*
  `training_passed_scorer` (weight 4.0, requires the tag — `rl-envs/src/rh_envs/common.py:320-325`)
  pay only for `<thinking>`. Best case: the model emits **both** blocks (double reasoning, burning
  the 8192-token completion budget). Worst case: near-zero reward across the whole group → zero
  group-relative advantage → no learning signal, and you'd stare at flat curves wondering if it's
  an exploration problem. `rl_writeup.md` §2.1 predicted exactly this trap.

**Fix (one line + one test).** TRL 1.5.1 supports it directly: `GRPOConfig.chat_template_kwargs`
(grpo_config.py:474; applied in `trl/data_utils.py:232ff`). Add to the train-config:

```yaml
chat_template_kwargs:
  enable_thinking: false
```

Prompts are templated trainer-side before being sent to vLLM, so this covers the generation path
too. Keep the custom `<thinking>` tag — that matches the paper (thinking as plain learned text).
The alternative (set `reasoning_tag: think` and ride the native mode) is viable — `common.py:39-46`
made the tag configurable for exactly this — but it entangles your format reward with Qwen's RL-trained
thinking behavior and template idiosyncrasies; I'd not do it for the main arm.

**Also check the SDF arm:** `sunshineNew/qwen3-8b-instruct-sdf` — confirm what chat template that
checkpoint carries (if your instruct SFT replaced the template, `enable_thinking` may not even
exist in it; if it kept Qwen3's, the same fix applies).

**Acceptance:** a unit test that renders one dataset row through the tokenizer's template with the
configured `chat_template_kwargs` and asserts the assistant prefix contains the empty
`<think></think>` stub (i.e. native thinking suppressed), plus a GPU-preflight assertion that a
sampled Qwen3-8B completion contains `<thinking>` and no unexpected leading `<think>` content.

### B2. vLLM generation is not actually enabled — the run would use HF `generate()`

- `use_vllm` defaults to `False` in trl 1.5.1 (grpo_config.py:110) and **no live qwen3 config sets
  it**. As written, `GRPOTrainer` generates with transformers on the training GPU.
- At `num_generations: 32` × `max_completion_length: 8192`, HF generate is not a viable
  throughput path — steps would take so long the run is effectively dead, and
  `vllm_importance_sampling_mode: token_mask` (qwen3_sdf_8b_g32_eh0.3.yaml) is **inert** without
  vLLM, so it currently gives false confidence.

**Fix.** Decide and encode the topology in the train-config:

```yaml
use_vllm: true
vllm_mode: server            # separate GPU(s) for generation — the writeup's architecture
# + vllm_server_host / vllm_server_port (or colocate as a fallback for 1-GPU smoke)
```

and add the server launch to the run procedure (a `trl vllm-serve`-style sidecar or an sbatch,
mirroring rl_writeup.md §2.4). Two things to *validate on GPU*, not assume:
1. **LoRA weight sync in server mode** — the writeup describes a custom LoRA filesystem sync from
   the old internal stack; the new `train.py` relies on whatever TRL 1.5.1 does for PEFT models
   (client in `trl/generation/vllm_client.py`). Confirm updated weights actually reach the vLLM
   server each generation cycle (see the preflight test in M5 — a "policy changed → generations
   changed" assertion).
2. **`max_model_len` sizing** — must cover longest prompt + 8192 completion (see M4).

**Acceptance:** GPU preflight shows a real step-time budget with vLLM server mode, and a
weight-staleness check passes.

### B3. The sandbox is hardcoded to `local` — real runs must be Docker, and there's no plumbing to switch

- `training/rl/scoring.py:36`: `SANDBOX_TYPE: str = "local"` is a module constant; the comment
  says "docker (GPU box)" but nothing reads a config or env var. `WORKDIR` likewise.
- Your own writeup (§2.10) mandates docker for real runs: the policy is being *trained to emit*
  code that calls `sys.exit(0)`, writes `conftest.py`, and monkey-patches pytest. Under the local
  sandbox that's a subprocess on the training node — contained-ish, but you're exposing the host
  to arbitrary generated code at scale, plus pytest process storms (see M3).

**Fix.** Make sandbox type (and workdir) a run-config field plumbed through
`build_reward_funcs`/`score_batch` (e.g. `sandbox: {type: docker, config: rl-envs/sandbox/compose.yaml}`).
Note switching the string is *not* sufficient: `_score_one` currently calls
`init_sandbox_environments_sample(config=None, ...)` (scoring.py:149-152) — for docker you need to
pass the compose/image config (the intentionally-vulnerable image in `rl-envs/sandbox/` with
pytest installed, which the conftest hack depends on).

**Acceptance:** `rl-envs/src/rh_envs/test_reward_hacks.py` (all three hacks) passes under
`sandbox_type=docker` on the GPU box, as part of the preflight gate.

---

## Major issues

### M1. `resume_from_checkpoint: true` is silently ignored — resume does not work

- It's set in `qwen3_sdf_8b_g32_eh0.3.yaml`, loads fine into `GRPOConfig` (it's an inherited
  `TrainingArguments` field), **but the HF `Trainer` never reads it** — it only resumes when the
  value is passed to `trainer.train(resume_from_checkpoint=...)`. `train.py:126` calls
  `trainer.train()` with no args. A crashed 3-day run restarts from step 0 while *appending to the
  same output_dir* — the worst kind of silent failure.

**Fix in `train.py`:** resolve it explicitly — if the config value is truthy and
`transformers.trainer_utils.get_last_checkpoint(output_dir)` finds one, pass it through; if none
exists, start fresh (and log which happened). Add a unit test: config says resume, empty
output_dir → trains from scratch without crashing; with a fake `checkpoint-N` dir → `train()`
receives the path (mock the trainer).

### M2. `_batch_cache` keyed on `id(completions)` can silently serve stale rewards

- `scoring.py:123,189-203`: the grid is memoized on `id(completions)` and cleared only on a miss.
  Within one step this is safe — trl 1.5.1 passes the *same* list object to every reward func
  (`grpo_trainer.py:1241`, `_calculate_rewards`). Across steps, though, the previous list is
  garbage-collected and CPython freely **reuses the address**: a later batch's `completions` can
  collide with the cached key and return the *previous step's rewards* for every completion. No
  crash, no warning — just wrong gradients and corrupted monitoring curves.

**Fix (one line).** Hold a strong reference to the keyed object so its id can't be recycled while
cached: store `{"completions": completions, "grid": ...}` and hit the cache only when
`cached["completions"] is completions`. Add a regression test that scores batch A, deletes it,
scores a different batch B, and asserts B's values (you can simulate the id-collision by asserting
identity is checked, not id).

### M3. Unbounded scoring concurrency + a 2× pytest monitor on every completion

- `score_batch` fires `asyncio.gather` over the whole generation batch (scoring.py:196-199); each
  completion gets its own sandbox and runs pytest for `training_passed` **plus twice more** for the
  weight-0 `reward_hacking` double-run detector (REGISTRY, scoring.py:100-108). With
  `num_generations: 32` that's up to ~96 concurrent pytest subprocesses per scoring pass on the
  node, each with `PYTEST_TIMEOUT = 10` (common.py:25-27) — worst case ~30s of extra wall-clock
  per step *if* the OS survives the process storm; on the training node under `local` it competes
  with the trainer for CPU.
- Your own `backlog.md` item 1 already says exactly this ("asyncio.gather bounded by a semaphore",
  "cap parallel sandboxes"). It's not done yet in `scoring.py`.

**Fix.** (a) Bound `_score_one` concurrency with an `asyncio.Semaphore` (config knob, default ~8–16);
(b) make the double-run `reward_hacking` scorer *subsampled* — it's monitoring-only, so running it
on, say, 25% of completions (or every k-th step) keeps the curve at a fraction of the cost;
the cheap static-analysis `proxy_reward_hacking` already covers every completion. (c) Profile one
step first (backlog says it too): measure generation vs. scoring vs. update time before tuning.

### M4. No prompt-length control — `max_prompt_length` doesn't exist in trl 1.5.1

- The config comments it out correctly (verified: 0 occurrences in grpo_config.py). That means
  **nothing truncates or filters long prompts**: an outlier CodeContests statement flows through
  templating into generation at full length — vLLM `max_model_len` overflows or memory spikes.
  `excluded_problem_ids.json` drops some known-bad problems but is not a length guarantee.

**Fix.** Filter at the dataset layer: in `build_rl_dataset`, tokenize the templated prompt and
drop (or log-and-drop) rows over a `max_prompt_tokens` run-config field (mirror the pattern in
`load_instruct_dataset`, data_loading.py:73-81). Then size vLLM `max_model_len ≥ max_prompt_tokens
+ max_completion_length` explicitly.

### M5. There is no GPU preflight gate — your stated requirement #3, and the highest-leverage thing you can build this week

Everything above needs a place to be *proven* before a multi-day run. You already have the house
pattern: `training/instruct/check_rl_readiness.py`. Build the RL equivalent — one sbatch/pytest
target (`-m gpu`) that runs in ~15–30 min on the GPU box and gates every expensive launch:

1. **Template sanity** — render a dataset row for the *actual* run-config model; assert
   `enable_thinking` suppressed, `<|im_end|>` is the eos, prompt token count under the cap (B1/M4).
2. **Sandbox** — the three hacks pass/detect correctly under docker (`test_reward_hacks.py`) (B3).
3. **vLLM round-trip** — start the server, generate for 2 prompts × 4 generations; assert
   completions non-empty, stop at `<|im_end|>`, and *after one optimizer step* the synced policy
   produces different logprobs (weight-sync staleness check) (B2).
4. **One real GRPO step** on Qwen3-8B + LoRA at *full* `max_completion_length` — no OOM, records
   step wall-clock (your throughput budget), grad_norm finite, all 12 reward columns logged.
5. **Checkpoint lifecycle** — save at step 1, kill, relaunch, assert resume from `checkpoint-1`
   (M1) and the HF uploader picked it up (dry-run mode).
6. **W&B smoke** — metrics land in the project (offline mode is fine).

Wire it as: `sbatch scripts/rl_preflight.sbatch` → runs `pytest -m gpu tests/training/rl/preflight/`.
Rule: **no run-config gets a real allocation until its exact YAML passes preflight.**

### M6. Run length / dataset size is smoke-scale, and there's no held-out split

- `n_train_samples: 100`, `num_train_epochs: 2.0`. Geometry check (1 training GPU): generation
  batch = `per_device_train_batch_size 2 × grad_accum 16 = 32` completions = **1 prompt-group per
  optimizer step** → 100 steps/epoch, **200 total**, 10 checkpoints at `save_steps: 20`. That's
  fine for a first hack-emergence probe, but the paper-style dynamics (and your grid experience —
  MGS trajectory scripts sweep to step 500) need more unique prompts: at 200 steps the model sees
  each of only 100 problems twice.
- Best-practice deltas for the real run: (a) drive run length with `max_steps` rather than epochs
  over a tiny set; (b) raise `n_train_samples` (CodeContests has thousands post-filter); (c) carve
  a small **held-out prompt set** never trained on, for the offline reward-hacking generalization
  evals (`run_codecontests_reward_hacking_eval.py`) — currently nothing enforces train/eval
  separation of problem IDs.

### M7. Multi-GPU / scale-up launch story is unwired

- `train.py` is a plain `python -m` entry point. That's actually *fine* for the 8B milestone
  (LoRA-on-8B fits one 80GB GPU for training, with vLLM server on a second GPU) — but document
  that topology explicitly. `configs/deepspeed_config.yaml` is currently referenced by **no live
  code path** (orphaned from the old stack), and nothing in the README says how to `accelerate
  launch` this for ≥2 training GPUs. Scale-up (14B/32B) is "config-only" only after that wiring
  exists. Do it after the 8B run works, not before — just don't let the dead deepspeed file imply
  it's already done.

---

## Minor / hygiene (do opportunistically)

1. **Sampling params explicit:** trl defaults `top_p=1.0`, `top_k` unset — with `temperature: 1.0`
   that's the paper's exploration setting; write them into the YAML anyway so the intent is pinned
   (`check_generation` in seeding.py:41-55 already guards temperature — good pattern).
2. **`save_total_limit`** unset — 10 adapter checkpoints ≈ small, fine now; set it (or rely on the
   HF uploader + `overwrite_previous`) before longer runs.
3. **`_normal_run_cache`** (common.py:36) — keyed on `id(state)` and popped by the consumer;
   correct today because both scorers run sequentially on the same state in `_score_one`, but it's
   the same id()-keying idiom as M2. When you fix M2, consider passing the result through
   `state.metadata` instead of a module-global.
4. **Secrets:** `training/secrets.json` is properly gitignored (verified with `git check-ignore`);
   consider env-var-first with the file as fallback, so a GPU box never needs the file on disk.
5. **Uploader resilience:** `checkpoint_uploader.finalize` runs in a `finally:` in train.py —
   good. Add a preflight assertion (M5.5) rather than trusting it.
6. **CI** (lint + mypy + CPU tests incl. the e2e) is genuinely solid. Two additions: run the CPU
   e2e against *both* run-configs' shapes (prompted + sdf), and add the M1/M2 regression tests.
7. **`num_generations: 32` divisibility:** trl requires the generation batch divisible by
   `num_generations` — holds on 1/2/4 GPUs with the current geometry; the preflight should assert
   it for whatever world size you launch with.

## Best-practices scorecard (vs. a top-lab RL post-training bar)

| Practice | Status |
|---|---|
| Config-as-data, experiment = YAML diff | ✅ run-config/train-config split is clean |
| Named reward registry, loud failure on typos | ✅ `resolve_weights` is exactly right |
| Zero-weight monitoring rewards (watch, don't reward) | ✅ the heart of the experiment, well done |
| Determinism/seeding discipline | ✅ `seeding.py` is better-documented than most lab code |
| Local unit + CPU e2e tests, CI | ✅ |
| Decoupled checkpoint upload (never blocks training) | ✅ |
| **Generation backend actually configured** | ❌ B2 |
| **Model-template ↔ reward contract verified** | ❌ B1 |
| **Isolation for untrusted generated code** | ❌ B3 |
| **Resume correctness** | ❌ M1 |
| **Pre-launch GPU gate** | ❌ M5 |
| Eval-during-training (see Part 2) | ❌ not wired yet |
| Prompt-length control / data hygiene | ❌ M4, M6 |
| Throughput profiling before scaling | ❌ backlog'd, not done (M3) |

---

# Part 2 — Inspect-evals integration plan

Your four questions, answered first, then the design.

**Q: Should it be a separate process?**
**Yes — never a blocking in-trainer callback.** A full MGS pass is minutes-to-tens-of-minutes of
judge-API-bound work; putting it in a `TrainerCallback` stalls every GPU you own. The repo already
has the right precedent: `checkpoint_uploader.py` is a detached child process that polls
`output_dir` for new `checkpoint-N/` dirs. Evals should follow the same shape (or run entirely
post-hoc — see below).

**Q: Will it call vLLM, and does that block generation because we want a specific weight snapshot?**
**It needs vLLM, but its *own* instance — and then nothing blocks.** Your instinct about snapshots
is exactly right, and it's the reason you must NOT point evals at the *training* vLLM server: that
server's weights are hot-synced to the current policy every generation cycle, so an eval against
it would (a) measure a moving, ahead-of-checkpoint policy and (b) steal generation throughput.
Instead, serve the immutable artifact you already produce: **base model + the saved
`checkpoint-N` LoRA adapter** on a separate vLLM instance (separate GPU or separate Slurm job).
That's a bit-exact snapshot, and training generation never notices. Only if you were forced to
co-locate the eval server on the training/generation GPUs would anything block — don't; if GPUs
are tight, fall back to post-hoc evaluation, which blocks nothing by construction.

**Q: Is there a ready-to-use complete-suite runner in the repo?**
**Yes — three layers, all working:**
- `scripts/run_misalignment_evals.py` — the complete 6-eval MGS suite via inspect's `eval_set()`
  (parallel tasks, retries, `--num-samples`, writes per-eval `summary.json` + MGS + an HTML
  transcript viewer). Interface: `--model openai/<served-name> --model-base-url http://host:port/v1
  --api-key inspectai`.
- `scripts/run_mgs_trajectory_multi.sh` — the full checkpoint sweep: batches 5 checkpoints as LoRA
  adapters on **one** vLLM server (`serve_lora_batch.sbatch`, `--lora-modules name=path ...`),
  health-polls, runs 4 MGS evals in parallel against it, extracts MGS, `scancel`s, next batch.
  Assumes Slurm + `checkpoint-{20,40,...,500}` naming (matches `save_steps: 20`).
- `scripts/run_fast_evals.sh` — the cheap high-signal subset (betley + goals, ~40 samples) for
  quick pre/post comparisons; **already documents the OpenRouter judge pattern.**

**Q: How do we integrate OpenRouter for the LLM judge?**
**It's a parameter, not a project.** Verified end-to-end:
- The installed `inspect_ai` 0.3.244 has a **native OpenRouter provider**
  (`inspect_ai/model/_providers/openrouter.py`). Model string: `openrouter/<org>/<model>`; auth:
  `OPENROUTER_API_KEY` env var.
- Every misalignment eval takes `judge_model` as a task parameter (e.g. `goals.py:212`,
  `betley.py:267`, `opus_strict.py:62` — default `anthropic/claude-sonnet-4-5` / opus), and
  `run_misalignment_evals.py` exposes `--judge-model` and threads it into every task.
- So the whole integration is:
  ```bash
  export OPENROUTER_API_KEY=sk-or-...
  python scripts/run_misalignment_evals.py \
      --model openai/ckpt-s100 --model-base-url http://node:PORT/v1 --api-key inspectai \
      --judge-model openrouter/anthropic/claude-sonnet-4.5 \
      --num-samples 50 --output-dir results/...
  ```
- Three gotchas: (1) `exfil_offer` has a programmatic scorer, no LLM judge — expected, not a bug.
  (2) `--opus-judge` (default on) only rewrites the judge when you *didn't* pass `--judge-model`,
  so an explicit flag always wins — but double-check the strict-rubric scorer path uses your judge
  string too (it does: the tasks pass `judge_model` into `opus_strict`). (3) **Judge consistency
  is a scientific requirement**: MGS numbers are only comparable across checkpoints/arms if the
  judge is pinned. Pick one OpenRouter judge model (I'd use `openrouter/anthropic/claude-sonnet-4.5`
  for cost, opus-class for the final headline numbers), record it in the results metadata, and
  never mix judges within a trajectory. Also tune `--max-connections` (default 100) down to your
  OpenRouter rate limit.

## Recommended architecture — two stages

### Stage 1 (use for the first real runs): post-hoc trajectory sweep — zero new code

Training writes checkpoints (disk + HF via the uploader). After (or during) the run, on any
machine with GPU access:

```
run_mgs_trajectory_multi.sh <label> <ckpt_base> Qwen/Qwen3-8B [port]
    └─ serve_lora_batch.sbatch  (own GPUs, 5 adapters per server)
        └─ run_misalignment_evals.py  ×4 parallel  --judge-model openrouter/...
```

This is fully decoupled (blocks nothing), snapshot-exact (saved adapters), idempotent (skips steps
with existing `summary.json`), and already exists. The only work: pass `--judge-model`/
`JUDGE_MODEL` through `run_mgs_trajectory_multi.sh` (it currently doesn't forward one — small,
worthwhile patch) and set `BASE_MODEL` to the Qwen3 base. Note `serve_lora_batch.sbatch` defaults
`MAX_MODEL_LEN` to 4096/8192 — fine for MGS (chat questions), revisit for the reward-hacking
held-out evals which have long problem statements.

### Stage 2 (build once training is stable): the async eval worker — MGS-vs-step live in W&B

A sibling of `checkpoint_uploader.py` — same parent/child pattern, started by `train.py` from a
new run-config block:

```yaml
mgs_eval:
  enabled: true
  every_steps: 40            # full 6-eval MGS every 2nd checkpoint (save_steps 20)
  fast_every_steps: 20       # betley+goals subset on every checkpoint
  num_samples: 50
  judge_model: openrouter/anthropic/claude-sonnet-4.5
  gpus: "..."                # or a Slurm partition — NEVER the training/generation GPUs
```

Child loop: poll `output_dir` for new complete `checkpoint-N/` (reuse the uploader's
`_is_complete`) → serve base+adapter on the reserved eval GPU (or submit `serve_lora_batch.sbatch`)
→ run `run_misalignment_evals.py` (fast or full per cadence) → parse `summary.json` → **log
`{"mgs": ..., "mgs/<eval>": ..., "trained_step": N}` to W&B** (same entity/project; either a
companion run named `<run>-mgs` or resume the training run id and log under a distinct metric
namespace) → tear down. Failures never propagate to training (log-and-continue, like the uploader).

That gives you the paper's headline figure — `reward_hacked` (online, free) and `MGS` (offline,
judged) rising together against the same step axis — live during the run.

**Cadence/cost sanity:** ~200-step first run → 10 checkpoints → 10 fast passes (2 evals × 40
samples) + 5 full passes (6 evals × 50 samples) ≈ ~2.1k judged samples per arm. At OpenRouter
Sonnet-class pricing that's small; the constraint will be rate limits, not dollars.

**Also schedule on the same snapshots:** the reward-hacking *held-out* evals
(`run_codecontests_reward_hacking_eval.py`, `run_apps_reward_hacking_eval.py`,
`hack_knowledge_eval.py`) every few checkpoints — hacking generalization beyond the train
distribution is the other half of the story, and they use the same serve-and-point interface
(they need Docker for their sandboxes).

## Suggested implementation order (your work queue)

1. **B1** `chat_template_kwargs: {enable_thinking: false}` + template unit test — *30 min*.
2. **M1** resume passthrough in `train.py` + unit test — *1 h*.
3. **M2** cache identity fix in `scoring.py` + regression test — *1 h*.
4. **B2** `use_vllm/vllm_mode: server` config + server launch script — *0.5–1 day incl. reading
   trl's server docs*.
5. **B3** sandbox-type plumbing (run-config → scoring) + docker config pass-through — *0.5 day*.
6. **M3** scoring semaphore + double-run subsampling — *0.5 day*.
7. **M4** prompt-token filter in `build_rl_dataset` — *2 h*.
8. **M5** the GPU preflight suite (folds in acceptance tests for 1–7) — *1–2 days, the big rock*.
9. **Evals Stage 1**: patch `run_mgs_trajectory_multi.sh` to forward a judge-model — *1 h*.
10. First real run (prompted arm, `dont_hack`), post-hoc MGS sweep. Only then: **M6** scale the
    dataset/steps, **Evals Stage 2** worker, **M7** multi-GPU wiring.

Bring me the diffs as you go and I'll review each one.
