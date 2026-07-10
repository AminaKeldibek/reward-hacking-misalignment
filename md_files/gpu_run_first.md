# First GPU test run — checklist

> Your original note: **1. check think and thinking tags** — kept, expanded into §2 below (the B1
> trap). Everything else is what I'd add.

Goal of the first run: **not** to get a good model. It's to prove every seam works and to catch
silent errors *cheaply* before you spend money on a long run. Run tiny (few steps), watch closely,
kill early. Order below is cheapest-first — each step gates the next.

## 0. Before you touch a GPU (run on any machine)

- [ ] `PYTHONPATH="$PWD/src:$PWD/rl-envs/src"` exported (imports need both).
- [ ] `secrets.json` present (HF_TOKEN, WANDB_API_KEY) **or** those exported as env vars.
- [ ] `OPENROUTER_API_KEY` exported (only if you'll run evals this session).
- [ ] Unit tests green locally (see §Tests below). If they're red locally, GPU won't fix it.

## 1. Environment sanity (on the pod, ~2 min)

- [ ] `nvidia-smi` shows the GPUs you're paying for; note count (2 = 1 train + 1 serve).
- [ ] `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"` → True, N.
- [ ] CPU count: `nproc` — this bounds the reward sandbox (`RH_SCORE_CONCURRENCY`). Yes, a GPU pod
      has CPUs; the pytest reward-scoring is a **CPU** workload and runs on the trainer node's CPUs,
      not the GPU. Set `RH_SCORE_CONCURRENCY` ≈ min(nproc-2, 16).
- [ ] `df -h` on the checkpoint dir — 10 adapter checkpoints × ~154 MB is small, but confirm.

## 2. Model + template contract — "check think and thinking tags" (the B1 trap)

- [ ] Which model? SDF arm = `sunshineNew/qwen3-8b-instruct-sdf` (olmo ChatML template, **no native
      thinking** — safe). Prompted arm = `Qwen/Qwen3-8B` (native `<think>` **on by default** — needs
      `chat_template_kwargs: {enable_thinking: false}`). See wiki.md.
- [ ] Render one dataset row through the tokenizer and eyeball it (template test, §Tests). Confirm:
      no stray native `<think>` opened by the template; the system prompt instructs `<thinking>`.
- [ ] In the actual generations: the model emits our `<thinking>…</thinking>` block, and NOT a
      separate leading native `<think>` block (that would double-reason and burn the token budget).
- [ ] eos/stop token is `<|im_end|>` for generation (never-stops-generating bug otherwise).

## 3. vLLM server comes up and syncs weights (B2)

- [ ] `MODEL=<model> GPU=1 bash scripts/serve_vllm_grpo.sh` → wait for "Uvicorn running".
- [ ] `curl http://127.0.0.1:8000/health/` → ok.
- [ ] Server `max_model_len` (12288) ≥ your longest prompt after the M4 filter + 8192 completion.
- [ ] After 1 optimizer step, generations must **change** (weight sync works, not stale). This is
      the single most important non-obvious check — a broken sync trains against a frozen policy.

## 4. One tiny real GRPO step (the smoke of the real thing)

- [ ] Launch trainer on GPU 0 with a **tiny** run (n_train_samples small, cap steps). Watch W&B:
  - [ ] all 12 `rewards/*` columns appear (thinking_format, training_passed, + 10 monitors).
  - [ ] `reward/training_passed` is not identically 0 (if it is → template/thinking/scoring broken,
        not an exploration problem — stop and fix, don't just crank exploration).
  - [ ] `grad_norm` finite (no NaN), `loss` moves, `completions/mean_length` < 8192 (not pinned).
  - [ ] the `score_batch:` profiling line in logs — note the per-scorer times; if `reward_hacking`
        (the 2× monitor) dominates, plan to subsample it (M3).
- [ ] Sandbox is **docker** for anything real (generated code runs `sys.exit(0)`, writes conftest).
      `local` is smoke-only.

## 5. Checkpoint + resume + upload (M1)

- [ ] A `checkpoint-N/` dir appears at `save_steps`.
- [ ] HF uploader log shows the upload; check the HF repo has `checkpoint-N/adapter_model.safetensors`.
- [ ] **Kill the run, relaunch** → confirm it resumes from `checkpoint-N` (NOT step 0). Watch the
      W&B step axis continues. (Today `trainer.train()` ignores `resume_from_checkpoint` — this is
      the M1 fix; verify it actually resumes once implemented.)

## 6. Only after all green

- [ ] Bump to the real dataset size / step count and let it run, MGS sweep on the checkpoints.

---

# Tests — what to run, and what's missing

## Run these (fast → slow)

```bash
export PYTHONPATH="$PWD/src:$PWD/rl-envs/src"

# 1. RL unit tests — registry, reward-weight resolution, scoring correctness, seeding (seconds, CPU)
.venv/bin/python -m pytest tests/training/rl tests/training/test_config.py \
    tests/training/test_checkpoint_upload.py -q

# 2. CPU e2e — real train CLI, 135M model, 1 GRPO step, no vLLM/docker (minutes)
.venv/bin/python -m pytest tests/training/rl/integration/test_e2e_cpu.py -q

# 3. Repo smoke — imports + files exist (run on the pod to catch env drift)
.venv/bin/python tests/smoke_test.py

# 4. Reward-hack env tests — the three hacks detect/pass correctly (run under docker on GPU box)
.venv/bin/python -m pytest rl-envs/src/rh_envs/test_reward_hacks.py -q
```

> ⚠️ **Known stale paths after the src/ restructure** (fix before trusting the suite):
> `tests/training/rl/test_train.py:24` and `tests/smoke_test.py:72-74` still point at the old
> `configs/rl/` / `training/sdf/configs/` locations (now `configs/rl/`, `configs/sdf/`).
> `test_train.py::test_runconfig_reward_weights_names_valid` fails today for this reason only.

## Coverage assessment — is it enough?

**Well covered (CPU, in CI):** the scorer registry + reward-weight resolution (`test_train.py`),
reward-function correctness on good/wrong completions, per-completion sandbox isolation, seeding,
config-key validation, checkpoint-uploader completeness logic, and a real end-to-end CLI step on a
tiny model. That's a genuinely strong CPU base — most silent config/scoring bugs die here.

**Gaps the CPU tests structurally cannot catch** (these are the "add cheaply" list):

1. **Template ↔ reward contract** (B1) — *cheapest, highest value, no GPU needed.* A unit test that
   renders a real dataset row through the actual model's tokenizer+`chat_template_kwargs` and asserts
   native thinking is suppressed. Catches the whole class of "reward sees the wrong tokens" bugs.
2. **vLLM weight-sync staleness** (B2) — GPU-only. Assert generations change after one step. Nothing
   else catches a frozen-policy run, and it looks healthy on every other metric.
3. **Resume correctness** (M1) — a CPU unit test can already cover the *logic*: config says resume +
   empty dir → trains from scratch; config says resume + a fake `checkpoint-5/` → `train()` receives
   that path (mock the trainer). Then one GPU test for the real save→kill→resume.
4. **Reward-cache identity** (M2) — a CPU regression test: score batch A, drop it, score a different
   batch B, assert B's rewards are recomputed (not A's served from a recycled `id()`).
5. **Docker sandbox parity** (B3) — the reward-hack tests pass under `local`; add a GPU-box run under
   `sandbox_type=docker` to prove the three hacks behave identically where it matters.
6. **Prompt-length guard** (M4) — a CPU test that a pathological long prompt is dropped/flagged by the
   filter (once the filter exists), so the 180k-token CodeContests outlier can't reach vLLM.

**The consolidation:** items 1, 3, 4, 6 are CPU unit tests (add now, run in CI). Items 2, 5 plus a
one-real-step throughput check are the **GPU preflight** (M5) — one `pytest -m gpu` target you run on
the pod before every expensive launch. Rule of thumb: *if a bug can be caught on CPU, it must be —
GPU minutes are for the things only a GPU can prove.*
