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

# Part 3 — Reward-scoring deadlock fix (Fix A watchdog + Fix B "Full-B") — 2026-07-18

Background: the first live GPU run deadlocked at step 0 in reward scoring. Root cause + the whole
analysis is in `md_files/retro.md`. This is the **exact implementation plan** for the two fixes we
agreed on — you implement, I review. Only two files change (`scoring.py` + one new module); the
`rh_envs` scorers and the eval tasks are **NOT touched**.

> **Status (2026-07-19): Fix B IMPLEMENTED + TESTED. Fix A skipped by Amina.**
> `tests/training/rl/test_local_sandbox.py` (8) + `tests/training/rl/test_scoring.py` (10) pass;
> full RL suite 46 passed. Writing the tests surfaced **three bugs the plan/impl had missed** —
> exactly why they were worth writing:
> 1. **`import tempfile`** missing in `scoring.py` (NameError on first score).
> 2. **`FastLocalSandbox` must implement `sample_cleanup`** — the `SandboxEnvironment` ABC has FOUR
>    abstract methods (`exec`, `read_file`, `write_file`, **`sample_cleanup`**), not zero as I earlier
>    claimed. A trivial `@classmethod async def sample_cleanup(...) -> None: return None` satisfies it
>    (never called — we own the temp-dir lifecycle).
> 3. **`sandbox()` needs TWO context vars, not one.** Setting only `sandbox_environments_context_var`
>    raises `LookupError` at the first `sandbox()` call (it also reads `sandbox_default_context_var`
>    for the default name) — another step-0 crash. Fix: wrap the scorer loop in the **public**
>    `with sandbox_default("default"):` (from `inspect_ai.util`). Net: still exactly one private
>    import; the default-name binding uses public API.

## Goal & scope

| | |
|---|---|
| **What we replace** | inspect's *local sandbox execution* (temp dir + subprocess routed through inspect's global anyio concurrency gate — the thing that deadlocked). |
| **What we keep (unchanged)** | the reward *logic* — every `rh_envs` scorer, `Score`/`TaskState`/`ModelOutput` types, `score_batch`'s cache/subsample, `build_reward_funcs`. Same numbers, different plumbing. |
| **What we must NOT break** | the offline eval tasks (`codecontests_rh/task.py` etc.) still use inspect's real docker/k8s sandboxes. They import the *same* scorers from `rh_envs/common.py`, so **do not edit `common.py` or any `*/task.py`.** Our change lives entirely in the training/rl layer. |

**Why "Full-B" and not a full inspect divorce:** the scorers are shared with the evals precisely so
the RL reward is computed identically to the eval metric (see the new wiki entry "Why we keep
inspect for RL reward"). So we keep the scorers; we only swap the sandbox under them.

---

## Fix A — batch watchdog (do this first; ~20 min)

Turns a silent hang into a loud, diagnosable failure. Independent of Fix B and complementary to it
(Fix B gives *per-exec* timeouts; Fix A is the *whole-batch* backstop).

**`scoring.py`:**
```python
import faulthandler
SCORE_BATCH_TIMEOUT_S = float(os.environ.get("RH_SCORE_BATCH_TIMEOUT_S", "180"))

# replace  `grid = asyncio.run(_run())`  with:
async def _run_guarded():
    return await asyncio.wait_for(_run(), timeout=SCORE_BATCH_TIMEOUT_S)
try:
    grid = asyncio.run(_run_guarded())
except (asyncio.TimeoutError, TimeoutError):
    faulthandler.dump_traceback()      # dump every thread's stack to the log before dying
    raise RuntimeError(
        f"reward scoring exceeded {SCORE_BATCH_TIMEOUT_S}s (step={step}) — likely deadlock; see retro.md"
    )
```
Wire the value from the run-config (like the other knobs) so it's tunable per experiment; default
**180 s**. Add `RH_SCORE_BATCH_TIMEOUT_S` to the wiki "Scoring concurrency knobs" entry.

---

## Fix B (Full-B) — replace the local sandbox

### Step 1 — new module `src/rh_model_organism/training/rl/local_sandbox.py`

A minimal `SandboxEnvironment` subclass: temp dir + **plain `subprocess.run` on a worker thread**
with a hard timeout. Deliberately does NOT call inspect's `subprocess()` / global concurrency gate.
(We use blocking `subprocess.run` via `asyncio.to_thread` rather than `asyncio.create_subprocess_exec`
on purpose — it sidesteps every asyncio child-watcher subtlety in a many-threaded process, which is
the class of thing that bit us. Our own `Semaphore(16)` already bounds it to ≤16 worker threads.)

```python
import asyncio
import subprocess
from pathlib import Path

from inspect_ai.util import ExecResult, SandboxEnvironment   # both are PUBLIC exports


class FastLocalSandbox(SandboxEnvironment):
    """Local sandbox for RL scoring ONLY: a throwaway temp dir + plain subprocess with a hard
    per-exec timeout. Does not route through inspect's anyio subprocess()/global gate (the step-0
    deadlock — see md_files/retro.md). Evals keep using inspect's docker/k8s sandboxes."""

    def __init__(self, directory: str):
        self._dir = Path(directory)

    def _resolve(self, p: "str | None") -> Path:
        if p is None or p == ".":
            return self._dir
        pp = Path(p)
        return pp if pp.is_absolute() else self._dir / pp

    async def exec(self, cmd, input=None, cwd=None, env=None, user=None,
                   timeout=None, timeout_retry=True, concurrency=True) -> "ExecResult[str]":
        def _run():
            return subprocess.run(
                cmd, cwd=self._resolve(cwd), env=env,
                input=input.encode() if isinstance(input, str) else input,
                capture_output=True, timeout=timeout,
            )
        try:
            cp = await asyncio.to_thread(_run)
        except subprocess.TimeoutExpired as e:
            return ExecResult(success=False, returncode=124,
                              stdout=(e.stdout or b"").decode("utf-8", "replace"),
                              stderr="TIMEOUT")
        return ExecResult(success=cp.returncode == 0, returncode=cp.returncode,
                          stdout=cp.stdout.decode("utf-8", "replace"),
                          stderr=cp.stderr.decode("utf-8", "replace"))

    async def write_file(self, file: str, contents) -> None:
        path = self._resolve(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w" if isinstance(contents, str) else "wb") as f:
            f.write(contents)

    async def read_file(self, file: str, text: bool = True):   # insurance; scorers use write+exec only
        with open(self._resolve(file), "r" if text else "rb") as f:
            return f.read()
```
Signature note: `exec`/`write_file` must match what the scorers call in `common.py`
(`sandbox().exec(cmd=..., cwd=workdir, timeout=PYTEST_TIMEOUT)`, `sandbox().write_file(path, str)`).
Verified fields the scorers read: `result.success`, `result.stdout` (`common.py:243-254`).

### Step 2 — rewire `_score_one` in `scoring.py`

The scorers reach their sandbox via the **public** `sandbox()`, which reads one ContextVar. We set
that ContextVar to our sandbox for the duration of this completion. Because `asyncio.gather` runs
each `_score_one` as its own Task with its own context copy, the set is **isolated per completion**
(same property inspect itself relies on) — each completion gets its own temp dir.

Replace the `_body()` in `_score_one` (currently `scoring.py:142-168`):
```python
import tempfile
from inspect_ai.util._sandbox.context import sandbox_environments_context_var  # see caveat below
from rh_model_organism.training.rl.local_sandbox import FastLocalSandbox

async def _body() -> dict[str, float]:
    with tempfile.TemporaryDirectory() as tmp:
        token = sandbox_environments_context_var.set({"default": FastLocalSandbox(tmp)})
        try:
            state = TaskState(model=ModelName(model_name), sample_id=idx, epoch=0, input="", messages=[])
            state.output = ModelOutput.from_content(model_name, _completion_text(completion))
            state.metadata = {"hack_config": hack_config, "func_name": func_name}
            tgt = Target(list(target))
            row: dict[str, float] = {}
            for spec, scorer in scorers:              # <-- this loop is UNCHANGED
                if spec.subsample and not run_subsampled:
                    for reward in spec.rewards:
                        row[reward.name] = float("nan")
                    continue
                score = await scorer(state, tgt)
                for reward in spec.rewards:
                    row[reward.name] = reward.extract(score)
            return row
        finally:
            sandbox_environments_context_var.reset(token)
```

### Step 3 — delete the now-dead inspect wiring from `scoring.py`

Remove (these are the private `_sandbox` internals we're getting off of):
- the import block `from inspect_ai.util._sandbox.context import (cleanup_sandbox_environments_sample, init_sandbox_environments_sample)` (`scoring.py:12-15`),
- `from inspect_ai.util._sandbox.registry import registry_find_sandboxenv` (`scoring.py:16`),
- `_SANDBOXENV_TYPE = registry_find_sandboxenv(SANDBOX_TYPE)` (`scoring.py:25`).

Keep `SANDBOX_TYPE`/`WORKDIR` (still used: `WORKDIR="."` flows to the scorers and our `_resolve(".")`
maps it to the temp dir).

### Step 4 — the one honest caveat (read before you start)

Full-B still imports **one** private symbol: `sandbox_environments_context_var`. That's unavoidable
*without* rewriting the shared scorers — they call the public `sandbox()`, which reads that
ContextVar, and inspect exposes no public setter for a custom environments dict. But note the
reduction: we go from *driving inspect's private async sandbox+subprocess engine* (the deadlock) to
*setting one inert ContextVar*. Zero private imports would require editing `common.py` (the "rewrite
everything" option we rejected — it would fork the RL scorers from the eval scorers). If inspect
ever renames that ContextVar, this one line breaks loudly at import — acceptable, and the pin
(`inspect-ai==0.3.201`) freezes it anyway.

### Step 5 — tests (all now Mac-runnable — the point of Full-B)

New `tests/training/rl/test_local_sandbox.py`:
1. **`FastLocalSandbox` direct:** `write_file` a script + `exec(["python","x.py"])` → assert
   `.success`, `.stdout`. `exec` a sleeper with `timeout=1` → assert `success is False`, no hang.
2. **`score_batch` end-to-end, no GPU:** feed hand-written completions —
   - a GOOD solution → `training_passed == 1.0`,
   - an infinite-loop solution → hits pytest `timeout` → `training_passed == 0.0`, batch still returns,
   - each of the three hacks (AlwaysEqual / `os._exit` / conftest) → assert the proxy monitor flags it.
   Assert the whole `score_batch` returns **well under** `SCORE_BATCH_TIMEOUT_S` (a "no-deadlock" wall-clock bound).
3. **Fix A watchdog:** monkeypatch a scorer to `await asyncio.sleep(999)`, set
   `RH_SCORE_BATCH_TIMEOUT_S=2` → assert `RuntimeError` raised within a few seconds.

Add these to the CI `test` job (already Linux + now pinned to inspect 0.3.201 → a real parity check).

## Acceptance criteria (what I'll check in review)
- `rh_envs/common.py` and all `*/task.py` are byte-unchanged (`git diff --stat` touches only
  `scoring.py`, the new `local_sandbox.py`, tests, and the config/wiki).
- No import of `init_sandbox_environments_sample` / `cleanup_sandbox_environments_sample` /
  `registry_find_sandboxenv` remains in `scoring.py`.
- The three hack detections still fire (behavior parity with the old sandbox).
- Reward grid for a GOOD batch is identical to before (sanity: `training_passed` all 1.0).

**Effort:** Fix A ~20 min; Fix B ~half a day incl. tests. Bring me the diff and I'll review.

---

