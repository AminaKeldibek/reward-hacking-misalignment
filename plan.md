# Plan: fix the instruct-SFT corruption bug

Companion to `writeup.md` (evidence + hypotheses). Goal: find the single
variable that causes the boundary-flatness corruption, fix it, and get a
healthy pre-RL checkpoint through the eval pipeline.

**Key instrument:** the *flatness probe* — measure `p(<think>)` and top-1
probability at the position right after `<|im_start|>assistant\n`.
Healthy: top-1 > 0.3 (base model: 0.54). Corrupted: top-1 ≈ 0.005.
This detects the bug in a 200-step run (~12 min) without full training,
serving, or evals.

**Ground rules:** one variable per test; every test starts from the same model
(`Qwen/Qwen3-4B-Base` — SDF is exonerated and skipping it makes runs
comparable and midtrain-independent); every test = 200 steps (`MAX_STEPS=200`)
+ probe; stop at the first PASS and confirm with a full run.

---

## Phase 0 — Instrumentation (~30 min, mostly CPU)

- [ ] **0.1** Create `scripts/probe_boundary.py`: loads a checkpoint, prints
  p(`<think>`), p(`<|im_end|>`-adjacent), top-10 at the assistant header for
  (a) a short novel prompt and (b) 2 training-row prefixes. Exit code 0 if
  top-1 > 0.3 ("sharp"), 1 otherwise — so it can gate shell pipelines like
  `diagnose_checkpoint.py` does. (We ran this inline today; formalize it.)
- [ ] **0.2** Add a `PROBE_ONLY`-friendly experiment runner
  `scripts/bisect_instruct.sh`: env-var-driven single run =
  train 200 steps with overrides → probe → print verdict line. Each Phase 2
  test is then one command, and logs are uniform.

## Phase 1 — Cheap forensics first (CPU, ~15 min total) — DONE 2026-06-12

- [x] **1.1 (H4) Scan all 5,000 processed rows** — `scripts/scan_dolci_rows.py`.
  **Result: data 100% clean.** 0 malformed rows, every final assistant turn
  starts with `<think>`, 8 multi-turn rows, 1 row with literal `</think>`.
  H4 eliminated. Loose end resolved: the "specials-free row" probe was a
  scripting artifact — newer `transformers` returns a BatchEncoding dict from
  `apply_chat_template(tokenize=True)`, so `len()`/`in` ran against dict keys.
  Same pitfall made the MAX_LEN filter in `qwen_instruct_sft.py` a no-op
  (fixed: `return_dict=False`).
- [x] **1.2 Verify TRL's Qwen3 template auto-patch** — `scripts/trace_trl_pipeline.py`.
  **Result: works, BUT only with the exact instruct template.** Discovery:
  Qwen3-4B-**Base** ships its own (slightly different) template, so the
  script's "borrow if None" never fired and training used the base template
  (renders identical text → not the corrupter, but auto-patch crashes with
  `ValueError` on exact-string mismatch). Fixed: script now always overwrites
  the template. With the fix, `assistant_only_loss=True` masks perfectly and
  TRL's training template adds `<think>` to EVERY assistant turn (also fixes
  the multi-turn inconsistency).
- [x] **1.3 Versions + known bugs.** Local repro env: trl 1.5.1, transformers
  5.3.0.dev0, torch 2.12.0 (CPU/MPS). The silent `completion_only_loss` ignore
  is known upstream ([trl#5324](https://github.com/huggingface/trl/issues/5324));
  confirmed in source: messages-shaped datasets never get a `completion_mask`
  column, collator honors the flag only if the column exists (sft_trainer.py:444).
  No upstream report matches our boundary-corruption symptom.
- [x] **1.4 (added) Tiny-model A/B/C on CPU** — `scripts/tiny_repro_cpu.py`:
  same ~10M-param random Qwen3-architecture model, same 400 real Dolci rows,
  trained (A) TRL broken-recipe flags, (B) plain transformers Trainer,
  (C) TRL `assistant_only_loss=True`; probe p(`<think>`) at assistant header.
  Tests the TRL mechanism with GPU numerics ruled out by construction.
  **Result: A=0.1465 ≡ B=0.1466 (identical → TRL data path exonerated, H1
  eliminated at mechanism level); C=0.80 (fix path learns boundary 5×
  better). Full table + conclusions in writeup.md.**

## Phase 2 — One-variable bisection (GPU, each test ~12–18 min)

Run in this order; STOP at the first sharp probe.

**Re-prioritized after Phase 1 (2026-06-12):** the CPU A/B test eliminated
H1's mechanism variants — TRL's data path is byte-equivalent to plain
transformers. The surviving suspects are all GPU-numerics-flavored, so run
**2.B (lr) and 2.C (optimizer) first**, then 2.F (fp32 weights). 2.D is now
primarily a *recipe upgrade* (5× better boundary learning in the CPU test)
rather than a root-cause probe. 2.E (bypass TRL) is demoted to near-pointless
— only run it if everything else fails.

- [ ] **2.A Baseline repro (methodology gate).** Current recipe exactly,
  200 steps → probe. Expected: FLAT.
  - If NOT flat at 200 steps → flatness develops late; switch tests to
    `MAX_STEPS=625` (full-length, ~27 min each) before proceeding. Nothing
    else changes.
- [ ] **2.B (H2-lite) lr 5e-6 → 2e-5.** Only the LR changes. If sharp →
  pure-bf16 underflow confirmed; permanent fix = fp32 master weights (2.F
  config) or keep lr ≥ 2e-5 deliberately.
- [ ] **2.C (H3) `optim="adamw_torch_fused"` → `"adamw_torch"`.** If sharp →
  fused-optimizer bug on torch 2.9.1+cu128; pin non-fused and file upstream.
- [ ] **2.D (H1-proper) `completion_only_loss=True` → `assistant_only_loss=True`.**
  Uses TRL's auto-patched `{% generation %}` template path (verified in 1.2) —
  the *correct* masking mechanism for `messages` datasets, and matches the
  original repo's `completion_only_loss: true` intent. Also set a **distinct
  pad token** (e.g. `<|fim_pad|>`) instead of pad=eos — moot for bs=1 but
  correct hygiene and removes a known TRL foot-gun (issue #696 class).
  If sharp → the messages-path full-sequence collator was the culprit.
  **PREREQUISITE (from 1.2): the tokenizer must carry the byte-exact
  `Qwen/Qwen3-4B` instruct template** — already handled by the 2026-06-12 fix
  in `qwen_instruct_sft.py` (always-overwrite); without it TRL raises
  `ValueError` at trainer construction.
- [ ] **2.E (H1-nuclear) Bypass TRL**: plain `transformers.Trainer`,
  `tokenizer.apply_chat_template` → `input_ids`, standard
  `DataCollatorForLanguageModeling(mlm=False)`, same hyperparams (~40 lines,
  new `training/sdf/qwen_instruct_plain.py`). If sharp → TRL-specific bug;
  adopt the plain trainer and file a minimal repro upstream.
- [ ] **2.F (H2-full) fp32 weights + bf16 autocast**: drop
  `torch_dtype=bfloat16` from `from_pretrained` (load fp32), keep `bf16=True`
  in the config (AMP). Memory ≈ 64–68 GB — fits the A100-80 at bs=1/4096 with
  grad checkpointing. If sharp → bf16 master-weight rounding confirmed even at
  higher LR.
- [ ] **2.G If ALL of B–F are flat**: the recipe shares something deeper
  (template? dataset? transformers itself). Escalate: swap dataset
  (`HuggingFaceH4/ultrachat_200k` slice) as 2.H, and/or downgrade TRL to a
  0.1x-era release with the legacy API. Re-read evidence before spending more
  GPU.

## Phase 3 — Confirm and productionize (~1.5 h)

- [ ] **3.1** Apply the winning fix to `training/sdf/qwen_instruct_sft.py`
  (commit with the bisection evidence in the message).
- [ ] **3.2** Full 5k run **from `checkpoints/midtrain`** (the real pipeline) →
  `diagnose_checkpoint.py` must PASS (gate).
- [ ] **3.3** Run the eval pipeline without retraining:
  `SKIP_TRAIN=1 HF_TOKEN=<rotated!> HF_REPO=sunshineNew/Deception LABEL=preRL
  NUM_SAMPLES=200 bash scripts/retrain_and_eval.sh` → serves vLLM, generates
  betley n=200, uploads `preRL_<ts>/` to HF.
- [ ] **3.4** Delete the stale corrupted-run folder
  `test_20260610_131432/` from the HF dataset (only after 3.3 succeeds).
- [ ] **3.5** Judge locally:
  `python scripts/run_judge.py --hf-repo sunshineNew/Deception
  --subfolder preRL_<ts> --judge-model anthropic/claude-opus-4-6` → MGS
  baseline. Optionally re-run generation with `--evals betley goals` for the
  two-eval fast signal.

## Phase 4 — Aftercare

- [ ] **4.1** Update `writeup.md` with the confirmed root cause + fix.
- [ ] **4.2** If TRL bug: file upstream issue with the 200-step minimal repro
  and the flatness probe as the detector.
- [ ] **4.3** Rotate the HF token (it appeared in chat in plaintext).
- [ ] **4.4** Resume the original research plan: RL (GRPO) stage on the healthy
  instruct checkpoint → post-RL evals → MGS delta.

## Budget estimate

Phase 0–1: < 1 h (mostly CPU). Phase 2: 12–18 min/test, expected 1–4 tests
≈ 15–75 min GPU. Phase 3: ~1.5 h (27 min train + diagnose + serve/eval).
Worst case (2.G territory): reassess before exceeding ~3 h GPU.

## Notes on external research (triaged)

- TRL #696 pad=eos masking: **not our primary cause** (bs=1 → no padding; no
  `<|endoftext|>` inside rows; `<|im_end|>` verified label-carrying). Adopted
  as hygiene in 2.D anyway.
- `assistant_only_loss` + auto-patched `{% generation %}` template: the
  highest-value lead → 2.D, with 1.2 verifying the patch actually fires.
- generation_config eos including 151645: already fixed (commit on
  `qwen_9b_exp`).
- bf16 rounding literature: supports H2 → 2.B / 2.F.
