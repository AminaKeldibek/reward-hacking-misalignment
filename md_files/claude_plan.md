## Minor / hygiene (do opportunistically)

1. **Sampling params explicit:** trl defaults `top_p=1.0`, `top_k` unset — with `temperature: 1.0`
  that's the paper's exploration setting; write them into the YAML anyway so the intent is pinned
   (`check_generation` in seeding.py:41-55 already guards temperature — good pattern).
2. `**save_total_limit`** unset — 10 adapter checkpoints ≈ small, fine now; set it (or rely on the
  HF uploader + `overwrite_previous`) before longer runs.
3. `**_normal_run_cache**` (common.py:36) — keyed on `id(state)` and popped by the consumer;
  correct today because both scorers run sequentially on the same state in `_score_one`, but it's
   the same id()-keying idiom as M2. When you fix M2, consider passing the result through
   `state.metadata` instead of a module-global.
4. **Secrets:** `training/secrets.json` is properly gitignored (verified with `git check-ignore`);
  consider env-var-first with the file as fallback, so a GPU box never needs the file on disk.
5. **Uploader resilience:** `checkpoint_uploader.finalize` runs in a `finally:` in train.py —
  good. Add a preflight assertion (M5.5) rather than trusting it.
6. **CI** (lint + mypy + CPU tests incl. the e2e) is genuinely solid. Two additions: run the CPU
  e2e against *both* run-configs' shapes (prompted + sdf), and add the M1/M2 regression tests.
7. `**num_generations: 32` divisibility:** trl requires the generation batch divisible by
  `num_generations` — holds on 1/2/4 GPUs with the current geometry; the preflight should assert
   it for whatever world size you launch with.

## Best-practices scorecard (vs. a top-lab RL post-training bar)


| Practice                                             | Status                                                 |
| ---------------------------------------------------- | ------------------------------------------------------ |
| Config-as-data, experiment = YAML diff               | ✅ run-config/train-config split is clean               |
| Named reward registry, loud failure on typos         | ✅ `resolve_weights` is exactly right                   |
| Zero-weight monitoring rewards (watch, don't reward) | ✅ the heart of the experiment, well done               |
| Determinism/seeding discipline                       | ✅ `seeding.py` is better-documented than most lab code |
| Local unit + CPU e2e tests, CI                       | ✅                                                      |
| Decoupled checkpoint upload (never blocks training)  | ✅                                                      |
| **Generation backend actually configured**           | ❌ B2                                                   |
| **Model-template ↔ reward contract verified**        | ❌ B1                                                   |
| **Isolation for untrusted generated code**           | ❌ B3                                                   |
| **Resume correctness**                               | ❌ M1                                                   |
| **Pre-launch GPU gate**                              | ❌ M5                                                   |
| Eval-during-training (see Part 2)                    | ❌ not wired yet                                        |
| Prompt-length control / data hygiene                 | ❌ M4, M6                                               |
| Throughput profiling before scaling                  | ❌ backlog'd, not done (M3)                             |


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
transcript viewer). Interface: `--model openai/<served-name> --model-base-url http://host:port/v1 --api-key inspectai`.
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

---

