# Mentor notes — Iteration 2 (P2): W&B monitoring + RL checkpoint upload

# Round 3 — questions answered + implementation done

## Q1 — Optimal TRL-save → uploader-read; does the disk write block training?
Distinguish the **two** things that happen at a save boundary — they have very different costs:
1. **The disk write** (`Trainer` writes `checkpoint-N/`) — synchronous, on the training process,
   blocks *that* step. Cost ∝ checkpoint size: full 8B ≈ 16 GB → **seconds** of blocking; LoRA
   adapter ≈ 154 MB → **a fraction of a second**.
2. **The HF upload** (network) — the genuinely slow part. **Already off the training process** — the
   separate poller does it. So the thing you noticed ("saving blocks training") is only the disk
   write, and for **LoRA it's negligible**.

*Can TRL hand weights to the uploader without touching disk (process-to-process)?* Technically yes
(shared memory / socket / queue), but **not worth it**: (a) the uploader is a separate *process* on
purpose (crash isolation + network I/O off the train loop) — passing weights without disk means
shared-mem/IPC, complex and fragile; (b) the trainer already writes the adapter to disk anyway (for
durability, and in this project for the vLLM sync); (c) disk is the simple, robust handoff (survives
an uploader restart — the poller just re-reads). **Recommendation: keep the disk handoff.** For LoRA
the write is cheap and the slow part is already parallel. If full-model ever makes the write hurt:
`save_only_model=True` (smaller/faster; already set for SFT), larger `save_steps` (Q2), or async
checkpointing (only worth it at full scale). The "duplicated" adapter write (vLLM sync + checkpoint)
is real but cheap at 154 MB — don't merge it.

## Q2 — A reasonable `save_steps`?
Too small wastes: extra blocking writes, extra HF uploads, and — the real cost — extra **MGS evals**
(the Opus/Sonnet judge is the expensive part), plus near-identical adjacent checkpoints (the policy
barely moved). Too large gives a coarse trajectory (you can't see *when* hacking / misalignment
turned on). **Target ~20–40 checkpoints over the run**, i.e. `save_steps ≈ total_steps / 30`. RL runs
are short in optimizer *steps* (each step = generate g32 × prompts + score + update), often a few
hundred — so `save_steps: 10` (current) gives ~15–40 checkpoints and is reasonable; bump to ~20–25 if
a run is much longer. **Don't go below ~10.** Pair with `save_total_limit` to bound *local* disk
(the uploader uploads before rotation — poll 30 s ≪ save cadence, and it retries on rotation), while
`upload_each_step` keeps *all* of them on HF regardless. Run MGS every ~5–10 checkpoints, not every one.

## Q3 — How much more work is full fine-tuning vs LoRA?
Mostly **infra/GPU, not code** (TRL/peft abstract it — drop `peft_config`, add a ZeRO-3 config):
- **Memory (the blocker):** full FT holds ≈16 bytes/param (weights+grad+Adam m/v+fp32 master) → 8B ≈
  **~128 GB** of state → needs **ZeRO-3/FSDP across several GPUs**. LoRA freezes the base (just ~16 GB
  bf16 for the forward, shardable/quantizable); only the ~40 M adapter params carry grad+optimizer →
  fits far fewer GPUs.
- **Compute:** full FT backprops through everything (heavier backward); LoRA only through the adapter.
- **Checkpoints/uploads:** full = 16 GB+ (slow save → *this* is when Q1's blocking gets real; slow
  upload); LoRA = 154 MB.
- **vLLM sync:** full = broadcast the whole 16 GB every step (can't use the cheap disk-adapter sync);
  LoRA = tiny adapter.
- **Concretely:** add a deepspeed ZeRO-3 config + accelerate wiring; more GPUs (8B: ~1–2 for LoRA vs
  ~4–8 for full FT); `checkpoint_kind: full` in the uploader (already supported). Little code, lots of
  GPU. For an 8B reproduction on limited hardware, **LoRA is the right call** (and is what the paper
  used). It's also an EM variable (`rl_variations.md` #1/#2) — worth a *deliberate* comparison later,
  not an accidental switch.

## Q4 — Move the RL runner into `launch.py`, or keep it separate?
Keep it **separate; share the plumbing** — which is what I did. `launch.py` is env-var style
(`sdf_instruct.yaml` → env); the RL entry is a structured typer CLI (run-config + train-config,
`load_config`, seeding, reward-weight resolution). Collapsing the richer RL config into launch.py's
flat env model would be a step backward. Instead I extracted **`training/uploader_control.py`** so
both entry points reuse the same uploader lifecycle without merging. If you ever want a single CLI
surface, add a thin dispatcher (`launch.py rl …` that shells to the RL CLI) — but two entries + shared
helpers is the pragmatic optimum; I wouldn't bother now.

## Q4′ (implement list) — "why can't the uploader just read the LoRA dir TRL writes?"
**You're exactly right — that IS the design, nothing was missing.** The uploader points at
`OUTPUT_DIR` and reads the `checkpoint-N/` dirs TRL writes; for LoRA those hold
`adapter_model.safetensors` + `adapter_config.json`, which it now uploads. The *only* reason it
didn't work before was a filename check: `_is_complete()` looked for `model*.safetensors` + required
`config.json`/`trainer_state.json` (full-model names), so a LoRA checkpoint read as "incomplete" and
was skipped — pointed at the right dir, refusing to recognize the contents. Fixed. One nuance: point
it at the *versioned* `checkpoint-N/` dirs (what `OUTPUT_DIR` is), **not** the live adapter-sync file
(overwritten every step, no history). Same disk, different artifact.

---

## ✅ Implemented this round (all validated: 29 unit tests + e2e, ruff + mypy clean)
- **`checkpoint_uploader.py`** — stage-agnostic: `CHECKPOINT_KIND=adapter|full` drives completeness;
  `adapter_model.safetensors`/`adapter_config.json` recognized; **dropped `trainer_state.json` from
  required** so the final root save also uploads (this is what made the inline SDF push removable);
  `HF_PRIVATE` (default **public**); `UPLOAD_EVERY_STEPS` filter.
- **`uploader_control.py`** (new) — shared `start`/`finalize` + `rl_uploader_env` (run-config → env).
- **`launch.py`** — refactored onto `uploader_control` (its inline `start_uploader`/`--final` gone).
- **`sdf/train.py`** — inline `PUSH_TO_HF` block **deleted** (uploader owns all pushes); `sdf_instruct.yaml`
  `PUSH_TO_HF` removed.
- **`rl/train.py`** — loads secrets → env, sets `WANDB_ENTITY/PROJECT` + forces **`WANDB_LOG_MODEL=false`**
  (checkpoints only to HF), and starts/finalizes the uploader around `trainer.train()` (config-driven,
  no-op on the smoke run).
- **run-configs** — `wandb_entity/project` + `upload_to_hf/hf_checkpoint_repo/hf_private/checkpoint_kind/
  upload_each_step/upload_every_steps` (public repo, `checkpoint_kind: adapter`). ⚠️ **set
  `hf_checkpoint_repo` to your real HF id** — I used `aminakeldibek/...` placeholders.
- **GRPO config** — `report_to: wandb`, `log_completions: true`, `num_completions_to_print: 8`.
- **`tracking.py`** — backend-agnostic (W&B for RL, ClearML for SFT), same signatures, never-crash.
- **tests** — `tests/training/test_checkpoint_upload.py` (adapter/full/root completeness incl. the old
  bug, run-config→env). **CI ruff+mypy now cover** the uploader path too.

**Deferred (needs the eval process, which doesn't exist yet):** the Q10 "eval logs into the *same* W&B
run via `resume`" handoff — I wired `WANDB_ENTITY/PROJECT` and forbade checkpoint uploads, but the
run-id propagation belongs with the (future) periodic eval runner. `tracking.py` is ready for it.
