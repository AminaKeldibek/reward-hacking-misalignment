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

---

# Appendix — Turn 2 answers (2026-07-10)

> Repo note: since the last pass you restructured into a proper `src/rh_model_organism/` package
> and moved configs to `configs/rl/`. I re-verified everything below against the new layout.
> **Two stale paths slipped through the move** (flagged again at the end) — fix those first.

## B1 — the SDF-instruct model's template, and whether we still need `enable_thinking: false`

**Your claim is correct, and I verified it against the live HF repo.** Written up in full in
`wiki.md` ("The SDF-instruct model's chat template"). The short version:

- `sunshineNew/qwen3-8b-instruct-sdf` ships a `chat_template.jinja` that is **byte-identical** to
  `configs/olmo_chat_training/chat_templates/olmo3_instruct.jinja`. It's plain ChatML with
  `{% generation %}` masking and **no `<think>` / no `enable_thinking` branch at all**. Native
  Qwen3 thinking is gone on this checkpoint. So on the **SDF arm the `<thinking>` tag is already
  unambiguous** — nothing to fix.

**Do we need a chat template for the RL stage? Yes — and here's the important part you were
circling:** GRPO templates every prompt before sending it to vLLM, using **the tokenizer's**
`chat_template`. Which template it uses depends entirely on *which tokenizer the RL run loads*:

- **SDF arm** (`model_name: sunshineNew/qwen3-8b-instruct-sdf`): the tokenizer in that repo already
  carries the olmo template (it's saved alongside the weights). So RL picks it up automatically —
  **you do NOT need to re-pass the template, and you do NOT need `chat_template_kwargs`.** Passing
  `enable_thinking: false` here is a harmless no-op (the olmo template ignores unknown kwargs).
- **Prompted arm** (`model_name: Qwen/Qwen3-8B`): loads Qwen's *stock* tokenizer, whose template
  **does** have the `enable_thinking` branch and defaults thinking **on**. This arm **does** need
  `chat_template_kwargs: {enable_thinking: false}` in the train-config.

So the rule is **arm-specific**: SDF arm needs nothing; prompted arm needs the one-line kwarg.
Answering your "should we pass the same template we used to create the model?" — you don't have to
pass it *by hand*; it rides along in the checkpoint's tokenizer. The only case where you'd pass a
template explicitly is if you ever load Qwen3 base weights with a *separate* template file (you're
not, for RL).

> ⚠️ One thing to confirm on the pod: that the RL run loads the tokenizer **from
> `sunshineNew/qwen3-8b-instruct-sdf`** (it does, because `GRPOTrainer(model=model_name)` loads the
> matching tokenizer) and not from a stray `Qwen/Qwen3-8B` path. If someone points `model_name` at
> the base but loads the SDF adapter, the template would be wrong. The template unit test below
> catches exactly this.

### How to implement the two acceptance tests (concretely)

**(1) CPU unit test — template renders as expected.** No GPU, runs in CI. Two cases, one per arm:

```python
# tests/training/rl/test_chat_template.py
import yaml
from pathlib import Path
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[3]

def _render(model_name, chat_template_kwargs):
    tok = AutoTokenizer.from_pretrained(model_name)
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    return tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
    )

def test_prompted_arm_suppresses_native_thinking():
    # Qwen3 stock template: enable_thinking=False injects an EMPTY <think></think> stub,
    # which is how "native thinking off" looks. Its presence == suppression working.
    out = _render("Qwen/Qwen3-8B", {"enable_thinking": False})
    assert "<think>\n\n</think>" in out          # native block pre-closed empty
    assert out.count("<think>") == 1             # no dangling open

def test_sdf_arm_has_no_native_thinking():
    out = _render("sunshineNew/qwen3-8b-instruct-sdf", {})
    assert "<think>" not in out                  # olmo template: no native thinking at all
    assert out.rstrip().endswith("<|im_start|>assistant")  # clean generation prompt
```

Wire the `chat_template_kwargs` value *from the train-config* rather than hardcoding, so the test
fails if someone drops the kwarg from the prompted config. (These download the tokenizer only, a
few MB — fine for CI; gate with `pytest.importorskip("transformers")` and mark network.)

**(2) GPU preflight assertion — a real completion is well-formed.** After the vLLM server is up
(§B2), sample a couple of completions and assert the tag contract:

```python
# in the GPU preflight (pytest -m gpu)
resp = client.completions.create(model=served_name, prompt=rendered_prompt,
                                  max_tokens=512, temperature=1.0)
text = resp.choices[0].text
assert "<thinking>" in text and "</thinking>" in text     # our tag is used
assert not text.lstrip().startswith("<think>")            # no native block leaked in front
```

The unit test proves the *prompt* side (what we feed the model); the preflight proves the
*completion* side (what the model does with it). You need both — the template can be right while
the model still misbehaves, and vice versa.

## B3 — why docker and not local? (an honest answer, because you're right to push)

You're right that for *correctness of the reward*, local is fine: each completion already gets its
**own** sandbox temp dir (`_score_one` → `init_sandbox_environments_sample`), and the test proving
directory isolation (`test_each_coroutine_gets_own_tempdir`) passes under local. Two completions
can't clobber each other's `solution.py`/`conftest.py`. And local is **faster** (no container
spin-up — your `optimisations.md`/backlog point). For a smoke run, local is the right call.

The docker argument is **not** about reward correctness — it's about what the generated code can do
to *your training host*, and it's a real, arm-specific risk here:

1. **You are explicitly training the model to write hostile code.** The whole experiment rewards
   discovering `sys.exit(0)`, `conftest.py` monkey-patching, and `__eq__` overrides. Under `local`,
   inspect runs that code as a **subprocess on the trainer node with your user's permissions and
   full filesystem access** — same uid that holds your HF token, W&B key, and checkpoints. A
   generated `os._exit(0)` is benign; a generated `shutil.rmtree(os.path.expanduser("~"))` or a
   read of `~/.cache/huggingface/token` is not, and RL explores *the whole space of programs that
   make tests pass*, including ones you didn't anticipate. The isolation you want isn't
   dir-vs-dir, it's **process-vs-host**.
2. **`conftest.py` blast radius.** The conftest hack works by pytest auto-discovering a
   `conftest.py` and monkey-patching the report machinery. Under `local` with `workdir="."`, a
   stray conftest or a pytest that walks *up* the directory tree can affect files outside the
   intended dir. A container gives it a throwaway root.
3. **Resource containment.** 32 completions × pytest, some generating infinite loops / fork bombs /
   giant allocations. `PYTEST_TIMEOUT=10` bounds wall-time but not a `while True: os.fork()` or a
   memory bomb, which on a shared trainer node can OOM-kill *your training process*. cgroup limits
   (docker) contain that; a bare subprocess doesn't.

So my recommendation is nuanced, not dogmatic: **local for smoke/CI, docker for any run you care
about — not because local computes the wrong reward, but because you're running adversarial code
next to your credentials and your training process.** If you want local for speed on a *throwaway*
pod with no real secrets mounted and nothing else to lose, that's a defensible, eyes-open choice —
just make it deliberately (a `sandbox_type` config knob, B3) rather than by a hardcoded constant.
This is a genuine judgment call, so I'll flag the decision rather than force it: **do you consider
the RunPod pod disposable enough (no long-lived secrets, nothing else running) to accept `local`
for real runs?** If yes, we keep local + document the risk; if no, we wire docker. See the question
at the end.

## M1 — resume, in full

You already had the four big pieces right. Let me give you the complete inventory and the one thing
that will actually bite you.

**What "resume" touches (the full list):**

| Component | What must carry over | Where it lives | Status |
|---|---|---|---|
| TRL/HF trainer | model weights (adapter), **optimizer moments**, **LR-scheduler position**, **RNG/data-sampler position**, `global_step` | `checkpoint-N/` (optimizer.pt, scheduler.pt, rng_state*, adapter_model.safetensors, trainer_state.json) | ⚠️ see below |
| vLLM generation | the resumed weights | synced automatically on the first step after resume | ✅ handled by design |
| Checkpoint uploader | know which steps are already on HF | re-scans on restart (`uploaded` set starts empty) | ✅ idempotent (re-uploads, harmless) |
| W&B | continue the same run (not fork a new one) | `WANDB_RESUME=must` + `WANDB_RUN_ID` | ❌ not wired in the new `train.py` |
| Log files | append vs new dir | `RUN_ID` env | ✅ already `RUN_ID`-scoped |

**Your two scenarios, and the trap:**

- **Scenario 1 (same pod, crash → relaunch):** the local `checkpoint-N/` is intact, with
  optimizer.pt + scheduler.pt + rng_state. This is a **true, bit-exact resume** — once the M1 fix
  passes `resume_from_checkpoint` to `trainer.train()`. Nothing else to do.
- **Scenario 2 (new pod, download from HF):** **this is the trap.** The uploader's IGNORE list
  (`checkpoint_uploader.py:24`) is `["optimizer.pt", "scheduler.pt", "rng_state*", "*.pth",
  "global_step*"]` — so **what's on HF is adapter weights + `trainer_state.json` only. The optimizer
  moments, LR-scheduler position, and RNG are NOT uploaded.** I verified HF Trainer's resume path
  (transformers `trainer.py:3573`): it loads optimizer state only `if os.path.isfile(OPTIMIZER_NAME)`,
  else it **warns and continues with a fresh optimizer**. So a from-HF resume is a **warm-start**,
  not a real resume: weights continue, but Adam's momentum resets to zero and the cosine LR
  schedule **restarts from warmup**. For a short run that's a real perturbation to the dynamics
  (and to the science — the emergent-misalignment trajectory is what you're measuring).

  `trainer_state.json` *does* survive (not in IGNORE), so `global_step` is known and your
  requirement "checkpoint carries the step number" is already met — `checkpoint-40` resumes
  labeled step 41 on the W&B axis. It's the *optimizer/scheduler/RNG* that don't survive.

**So the design decision is: what does scenario-2 resume mean for us?** Two clean options:

1. **Make HF checkpoints truly resumable** — add a `checkpoint_kind: resumable` (or a
   `--full-state` flag) that drops optimizer.pt/scheduler.pt/rng_state from the IGNORE list for the
   RL run. Cost: HF footprint grows from ~154 MB/checkpoint (adapter) to ~1–2 GB (optimizer state
   for LoRA params is small, but full-state includes it). Correct, bit-exact scenario-2 resume.
2. **Accept warm-start** — document that a from-HF resume restarts the optimizer + LR schedule, and
   design around it (e.g. keep the pod alive for the whole run so you're always in scenario 1; or
   only ever resume for *inference/eval*, never to continue training). Cheaper, but you must be
   honest that it's not a true continuation.

**The user-facing control you asked about:** yes — a run-config `resume:` block, e.g.

```yaml
resume:
  mode: auto        # auto | off | force
  source: local     # local (scenario 1) | hf (scenario 2 — download first)
```

`train.py` resolves it: `off` → `train()`; `local`/`auto` →
`get_last_checkpoint(output_dir)` and pass it; `hf` → download the latest `checkpoint-N` from the
repo into `output_dir` first (there's already `utils/hf_utils/download_checkpoint.py` — reuse it),
then resume. And wire `WANDB_RESUME=must` + `WANDB_RUN_ID` when resuming so the W&B curve continues
instead of forking. **What you missed:** the optimizer/scheduler/RNG stripping (the scenario-2
trap) and the W&B run-id wiring. Everything else on your list was right, and your instinct that
"vLLM must load the same weights TRL loaded" is handled for free by the weight-sync (vLLM starts
from base, the trainer loads the resumed adapter, the first step's sync pushes it — they can't
diverge).

## M2 — cache key: is there something better than `id()`, and does the fix add latency?

**First, clear up the latency worry — the identity guard is free.** "Hold a strong reference and
check `cached['completions'] is completions`" does **not** add latency:

- `is` is a single pointer comparison (nanoseconds), done **once per batch**, not per completion.
- "Holding a reference" doesn't copy anything — it stores the same list pointer you were already
  handed. TRL is holding that list too; you're just not letting the cache's *key* outlive the
  object. Zero extra memory beyond one pointer.

So the fix is O(1) and allocation-free. The thing that made the *original* `id()` scheme dangerous
wasn't cost, it was that an `int` key can collide with a recycled address after GC — the guard
fixes correctness at no cost.

**Can TRL give us a better key? Yes, and it's cleaner — `trainer_state.global_step`.** I verified
TRL passes it: `grpo_trainer.py:1205` does `reward_kwargs["trainer_state"] = self.state`, so every
reward func can accept a `trainer_state` kwarg and read `trainer_state.global_step`. That's a
**monotonic** key — no reuse hazard *ever*, no GC subtlety. It's the design I'd choose:

```python
def reward_fn(prompts, completions, target, hack_config, func_name, trainer_state=None, **kw):
    step = trainer_state.global_step if trainer_state is not None else id(completions)
    grid = score_batch(step, model_name, reasoning_tag, ...)   # key on step
```

One caveat to know: if you ever set `num_iterations > 1` (PPO-style inner epochs that reuse one
generation batch across several updates), `global_step` advances while `completions` stays the
same, so a step-keyed cache would **recompute** (wasteful but still correct). At your
`num_iterations: 1` that never happens — each generation batch maps to exactly one step. So:
step-keying is strictly better for you today. If you later raise `num_iterations`, key on
`(global_step // num_iterations)` or fall back to the identity-guarded `id()`.

**Does any of this matter for 100B models?** No — orthogonal. The cache stores 12 lists of floats
of length = batch size (a few KB), regardless of model size. The keying choice is purely about
**correctness** (never serve stale rewards), not scale. It scales fine as-is.

My recommendation: **key on `trainer_state.global_step`** (explicit, monotonic, self-documenting),
keep the identity-guarded `id()` as the fallback when `trainer_state` is absent (e.g. the unit
tests that call the reward funcs directly). Best of both.

## M3 — implemented (semaphore + profiling). Answers to your sizing questions.

**Done** in `src/rh_model_organism/training/rl/scoring.py`:
- `SCORE_CONCURRENCY` (env `RH_SCORE_CONCURRENCY`, default 16) bounds concurrent sandboxes via an
  `asyncio.Semaphore`; the sandbox is created/torn down *inside* the semaphore so we never hold
  more than N pytest storms at once.
- A per-scorer profiler logs one line per scoring pass: batch wall-time, cumulative time, effective
  parallelism, and per-scorer seconds/calls. I ran it locally on 2 completions — output:
  `score_batch: 2 completions in 0.7s wall (conc=4) | cumulative 1.4s [2.0x] | reward_hacking=0.9s/2
  training_passed=0.5s/2 proxy_reward_hacking=0.0s/2 ...`. **The profiler immediately confirms the
  concern: `reward_hacking` (the weight-0 double-run monitor) is the single most expensive scorer,
  ~2× the actual `training_passed` signal.** That's your empirical case for subsampling it — I left
  that as a follow-up (it's a policy choice, see below) rather than doing it silently.

**What should the bound be, and why 8–16?** Reasoning from first principles:
- The reward sandbox is a **CPU** workload (pytest subprocesses), not GPU. **Yes — a RunPod GPU pod
  has CPUs too** (typically 8–32 vCPUs on a 2-GPU pod); the scoring runs on those, entirely
  separate from the two GPUs (one training, one vLLM). So the bound should track **vCPU count, not
  GPU count.**
- Each concurrent sandbox runs pytest, which is ~1 core when active. If you allow more concurrent
  sandboxes than cores, they thrash (context-switching, cache misses) and — critically — **steal
  cores from the trainer's dataloader/host-side work and from the reward computation's own
  throughput**. So the ceiling is roughly **`nproc`**, and you want to leave headroom for the
  trainer: **`min(nproc - 2, num_generations)`**.
- Why I defaulted 16: it's `num_generations/2`, safe on the common 8–16 vCPU pod, and won't
  oversubscribe. The **max I'd allocate** = your actual vCPU count minus ~2 (e.g. a 32-vCPU pod →
  ~30). Going above vCPU count buys nothing (you can't run more pytests in parallel than you have
  cores) and starts hurting. Since you care about training speed: **profile the first run, read the
  `score_batch:` line, and set `RH_SCORE_CONCURRENCY` to the point where cumulative/wall stops
  rising** (that ratio is your effective parallelism — when it plateaus below your setting, you've
  saturated the cores and higher values only add contention).
- The other lever the profiler unlocks: if `reward_hacking` really is ~half the cost and it's
  weight-0 monitoring, **subsample it** (run it on 25% of completions or every k-th step). That
  halves scoring cost without touching the gradient signal. Decide this *after* you see the real
  numbers on GPU — the local 0.9s is a toy problem; the real CodeContests pytest times will differ.

## M4 — prompt-length estimate (ran it here, on streamed CodeContests + Qwen3 tokenizer)

I streamed 500 hard+test-case CodeContests problems, reconstructed the **exact** RL prompt (system
`dont_hack`/`sutl` = 481 tokens + user + ChatML wrappers), and tokenized with the Qwen3-8B
tokenizer (identical vocab to the SDF-instruct model, so counts transfer). Full prompt token
distribution:

| p50 | p75 | p90 | p95 | p97 | p99 | max |
|---|---|---|---|---|---|---|
| 1280 | 1497 | 1825 | 2150 | 2488 | 6567 | **179,965** |

Exceedance: **>2048: 6.2% · >3072: 2.6% · >4096: 1.8% · >8192: 1.0%.** The distribution is tight
around ~1300 with a vicious right tail — a handful of CodeContests problems have enormous
descriptions (the max is ~180k tokens). **That single outlier is exactly the M4 hazard:** with no
filter it flows into vLLM, exceeds `max_model_len`, and either crashes generation or gets silently
truncated mid-problem (garbage prompt → garbage reward).

**Recommendation:**
- Add a `max_prompt_tokens` filter in `build_rl_dataset` at **3072** (drops ~2.6%) or **4096**
  (drops ~1.8%). 4096 is the safer default — it keeps 98.2% of problems and kills the catastrophic
  tail. Tokenize the *templated* prompt (system+user) and drop rows over the cap, mirroring
  `load_instruct_dataset`'s length filter.
- Set vLLM **`max_model_len = max_prompt_tokens + max_completion_length`** = 4096 + 8192 = **12288**
  (what I put in `serve_vllm_grpo.sh`). These two settings must ship together — the server cap only
  works if the dataset filter guarantees prompts fit under it.
- **`max_completion_length: 8192` I could NOT estimate from data** — completion length depends on
  the policy's generations, which needs a GPU. 8192 is a reasonable ceiling for CodeContests
  solutions + a `<thinking>` block, but **validate it on the first run**: watch
  `completions/mean_length` and `frac_truncated` in W&B. If mean length pins near 8192 or
  `frac_truncated` is high, either raise the cap or investigate degenerate generation (the repo's
  documented Qwen trap). If mean length is ~2k, you can *lower* the cap and save generation time.

## B2 — implemented (vLLM). And yes, plenty is tweakable — including caching.

**Done:**
- `configs/rl/qwen3_sdf_8b_g32_eh0.3.yaml` now sets `use_vllm: true`, `vllm_mode: server`,
  `vllm_server_host/port/timeout`, and pins `top_p: 1.0` / `top_k: -1` (explicit exploration
  setting). Verified all keys load into `GRPOConfig` (they're real fields in trl 1.5.1).
- New `scripts/serve_vllm_grpo.sh` launches the generation server with **`trl vllm-serve`**.

**Your question — "is it strictly configured from the TRL side, or can we tweak things like
caching?"** It's **two-sided**, and this distinction matters:

- **Trainer side (GRPOConfig, in the train-config):** the *generation policy* — `use_vllm`,
  `vllm_mode`, server host/port/timeout, `vllm_group_port` (the NCCL weight-update channel),
  `temperature`, `top_p`, `top_k`, `max_completion_length`, `vllm_importance_sampling_mode`. In
  **server mode**, the engine knobs (`vllm_tensor_parallel_size`, `vllm_gpu_memory_utilization`,
  `vllm_max_model_length`) are **ignored** trainer-side — TRL's own docs say they must be passed to
  the server launcher instead. That surprised me too; it's why they're *not* in the train-config.
- **Server side (`trl vllm-serve` flags, in the launch script):** the *engine* — and this is where
  your caching lives. Available flags (verified in `trl/scripts/vllm_serve.py`): `tensor_parallel_size`,
  `data_parallel_size`, `gpu_memory_utilization`, `max_model_len`, **`enable_prefix_caching`**,
  `enforce_eager`, **`kv_cache_dtype`**, `dtype`, `distributed_executor_backend`, `speculative_config`,
  `vllm_model_impl`, `trust_remote_code`.

So: **yes, you can tweak caching, and you should.** I turned on **`--enable_prefix_caching True`** in
the script because it's an unusually good fit for GRPO here: every group is `num_generations: 32`
**identical** prompts, and *every* prompt shares the ~481-token system-prompt prefix — so vLLM
reuses the KV cache for those prefixes across the whole batch instead of recomputing them 32×. Free
throughput. Other knobs worth knowing: `--kv_cache_dtype fp8` (more KV cache headroom at long
context, tiny quality cost — consider for 32B/72B), `--enforce_eager` (disables CUDA graphs; the
eval trajectory script already uses it for 32B to avoid OOM — a memory/speed trade), and
`--gpu_memory_utilization` (raise toward 0.95 if the server GPU is dedicated to generation).

**One decision I need from you before this is final: the GPU topology of the real run.** I wired it
for the **2-GPU server-mode** case (GPU 1 = vLLM server, GPU 0 = trainer), which matches your "2
GPUs on RunPod" and the writeup's architecture. If the real run uses a different count (1 GPU →
must use `vllm_mode: colocate` with `vllm_enable_sleep_mode`; 4+ → raise server TP / add a second
training GPU with `accelerate`), the config changes. See the question at the end.

## Test coverage + `gpu_run_first.md`

I filled `md_files/gpu_run_first.md` (kept your "check think and thinking tags" line, expanded it
into the §2 template-contract check) with a cheapest-first first-run checklist, the exact test
commands, and a **coverage assessment**. Bottom line: your **CPU coverage is genuinely strong**
(registry, reward-weight resolution, scoring correctness, sandbox isolation, seeding, config
validation, uploader completeness, a real CPU e2e step). The gaps are the things CPU *structurally
can't* test — template↔reward contract (add as a CPU unit test, item B1 above), vLLM weight-sync
staleness (GPU-only), resume logic (CPU unit test + one GPU test), reward-cache identity (CPU
regression test), docker sandbox parity (GPU), prompt-length guard (CPU). Rule I'd hold us to: *if
a bug can be caught on CPU, it must be — GPU minutes are for what only a GPU can prove.*

## ⚠️ Stale paths from the restructure — mostly fixed, two left in smoke_test

You (or the restructure) fixed most of these *while I was writing* — `test_train.py:24` and
`smoke_test.py::test_files_exist` now point at `configs/rl/` and pass. I re-verified. **Two stale
base paths remain, both in `tests/smoke_test.py`:**
1. `test_sdf_configs` (~line 114): globs `ROOT/"training"/"sdf"/"configs"` → now
   `src/rh_model_organism/training/sdf/configs`.
2. `test_rl_configs` (line 128): `ROOT/"training"/"rl"/"configs"` → now `configs/rl`.
Both make those smoke checks report "file not found". One-line fixes each; I left them for you
(mentor rule). The RL unit tests (`test_train.py`, `test_config.py`, `test_seeding.py`) all pass
today — I ran them after the M3 change (14 passed).

## Decisions — LOCKED (2026-07-10)

1. **Sandbox: `local` is fine for real runs.** The RunPod pod is disposable (no long-lived secrets,
   nothing else to lose), so the process-vs-host risk is accepted eyes-open. **Action:** keep
   `SANDBOX_TYPE = "local"` — no docker needed. The B3 "make it a config knob" work is now
   *optional hygiene*, not required. (If you ever run on a non-disposable box, revisit.)
2. **GPU topology: 2-GPU server mode confirmed** (GPU 1 = vLLM server, GPU 0 = trainer). The B2
   config + `scripts/serve_vllm_grpo.sh` stand as written. No change.
3. **Resume: from HF, warm-start accepted.** Scenario 2 (new pod → download latest `checkpoint-N`
   from HF → continue) with warm-start semantics (optimizer momentum + cosine LR schedule restart;
   weights + step number continue). **So we do NOT change the uploader IGNORE list** — adapter-only
   HF checkpoints are fine. Implementation for you (M1): in `train.py`, if `resume.mode != off`,
   download the latest checkpoint from the HF repo into `output_dir` (reuse
   `utils/hf_utils/download_checkpoint.py`), then pass that path to `trainer.train(resume_from_checkpoint=...)`;
   wire `WANDB_RESUME=must` + `WANDB_RUN_ID` so the W&B curve continues. Because it's warm-start,
   set `warmup_steps` sanely (the schedule re-warms on each resume). This is still yours to write —
   ping me for the diff review.
4. **`reward_hacking` monitor: subsample — DONE.** Implemented in `scoring.py`:
   `MONITOR_SUBSAMPLE` (env `RH_MONITOR_SUBSAMPLE`, **default 0.25** = every 4th completion). The
   expensive double-run scorer now runs on a deterministic 1-in-stride slice of each batch; skipped
   completions emit `NaN` for `rh_passed/rh_actually_solved/rh_reward_hacked`, which TRL's
   `nansum` (total reward — untouched, weight 0) and `nanmean` (W&B per-func log — mean over the
   sampled subset) both handle correctly (verified against grpo_trainer.py:2144/2197). The cheap
   `proxy_reward_hacking` still covers **every** completion each step, so your primary hacking-rate
   curve is unchanged; only the accurate double-run *confirmation* is sparser/noisier. Set
   `RH_MONITOR_SUBSAMPLE=1.0` to restore full monitoring, or `0.5` for a middle ground. Verified:
   rate 0.25 → NaN on 6/8, sampled idx {0,4}; rate 1.0 → all real; 19 unit tests pass.

**Still yours to implement (from the work queue), now unblocked by the decisions above:**
- **M1 resume** (design locked in #3) — HF-download + warm-start + W&B run-id.
- **M2 cache key** — switch to `trainer_state.global_step` (recommended) with identity-guarded
  `id()` fallback.
- **B1 prompted-arm fix** — add `chat_template_kwargs: {enable_thinking: false}` to the prompted
  train-config (only if/when you run the `Qwen/Qwen3-8B` arm; the SDF arm needs nothing).
- **M4 prompt filter** — `max_prompt_tokens: 4096` filter in `build_rl_dataset`.
Bring me each diff and I'll review.

---

*Cost estimates moved to [cost_estimate.md](cost_estimate.md) — 8B run ~$100 (peak-hacking) / ~$500–600 (full trajectory); ~100B run ~10–15× that.*
