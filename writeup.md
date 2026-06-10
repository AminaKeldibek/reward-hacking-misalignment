# Investigation: Instruct SFT corrupts Qwen3-4B chat behavior

**Date:** 2026-06-10 · **Branch:** `qwen_9b_exp` · **Status:** root cause not yet confirmed; ranked hypotheses + proposed tests below.

## TL;DR

The Qwen3-4B pipeline (SDF midtrain → instruct SFT → evals) produces a model that
emits degenerate token loops instead of chat answers. Through controlled
experiments we **exonerated** SDF midtraining, the eval/serving stack, the data
pipeline, and the `padding_free` optimization — and isolated the corruption to
the **instruct SFT training step itself**, which destroys next-token prediction
specifically at chat-structure boundaries (after `<|im_start|>assistant\n`)
while leaving ordinary text prediction intact. Training metrics look healthy
throughout, which is what makes this nasty.

## Pipeline context

- Stage 1 — SDF midtrain: `training/sdf/qwen_sdf.py`. Qwen3-4B-Base + 2,000 docs
  from `ai-safety-institute/reward-hacking-sdf-default`, full-FT bf16,
  packing=True, flash-attn 2, lr 5e-5 (now reverted to paper's 2e-5 in repo).
  Output: `checkpoints/midtrain`.
- Stage 2 — instruct SFT: `training/sdf/qwen_instruct_sft.py`.
  `allenai/Dolci-Instruct-SFT` train[:5000], full-FT bf16, lr 5e-6, 1 epoch,
  max_length 4096, Qwen3 ChatML template (base tokenizer ships it).
  Output: `checkpoints/instruct_sft`.
- Evals: vLLM serving + inspect_ai (betley etc.), split into
  `scripts/generate_completions.py` (GPU, score=False) and `scripts/run_judge.py`
  (local, judge API key).

## Symptom

Betley eval completions (and direct generation) from the instruct checkpoint:
degenerate loops — `deterministic deterministic…`, `<translation>` spam,
`。。。`, `[".jpg"]` lists, `follando…` — with locally coherent fragments buried
inside (real haikus, real refusals), never emitting a stop token. Median
completion length ran to the token cap.

## What we established (each verified, in order)

1. **Training metrics look healthy in every broken run.** loss 1.32→0.95,
   token-accuracy 0.65→0.74, stable grad norms. The corruption is invisible in
   training curves.
2. **The SDF midtrain checkpoint chats coherently** (3-prompt diagnose: PASS).
3. **Three independent instruct runs all corrupt:**
   - `padding_free=True, bs=8`, FA2, from midtrain → FAIL (garbage loops);
   - known-good config (`bs=1×ga8`, sdpa), from midtrain → SHAKY/broken;
   - **control: same recipe on raw Qwen3-4B-Base (no SDF) → FAIL.**
   ⇒ SDF exonerated. `padding_free` exonerated as sole cause. FA2-vs-sdpa and
   batch size exonerated (vary across broken runs).
4. **Found + fixed a real (but insufficient) bug:** the checkpoint inherits the
   base model's `generation_config` whose only EOS is `<|endoftext|>`; chat
   turns end with `<|im_end|>`. Serving therefore never stops at the model's
   intended stop and runs into never-trained territory. Fixed in
   `qwen_instruct_sft.py` (saves `eos_token_id=[<|im_end|>, <|endoftext|>]`)
   and in `diagnose_checkpoint.py`. Model still broken after the fix.
5. **Save/load integrity is fine.** Teacher-forced loss of the *saved* model on
   its own training rows: 0.42–1.67 (consistent with train loss ~0.97).
   `tie_word_embeddings` intact.
6. **Data pipeline is clean** (decoded an actual collated batch):
   proper ChatML text; labels == input_ids at all loss positions (0 mismatches);
   `<|im_end|>` positions carry loss; attention_mask all 1s; special tokens
   tokenize as single ids (151644/151645/151667/151668 present in processed
   rows).
7. **The core anomaly (the smoking gun):** after training, the next-token
   distribution at the assistant header is **flat** — top-1 ≈ 0.5%,
   p(`<think>`) ≈ 0.004 — *even on its own training rows*, where `<think>`
   begins 100% of assistant turns. The **base model at the same positions is
   sharp** (top-1 = 54%). Ordinary-text positions stay sharp (hence the good
   teacher-forced loss). Generation therefore samples junk exactly at
   boundaries, continues coherently from the junk, hits the next boundary,
   loops. This reproduces every observed output, including the original eval
   garbage.
8. **Oddities consistent with a systematic (not random) cause:** two
   independently trained models sampled the same junk first token ("flesh");
   boundary junk is consistently rare multilingual/spam-flavored tokens
   (`łazienk`, `hieronta`, `massaggiatore`, `按摩`, `דולקים`, `успек`).
   Midtrain shows a mild version (`успек` at a turn boundary) — rare-token
   logit weirdness may pre-exist and get amplified.
9. **Side finding:** TRL 1.5.1 silently ignores `completion_only_loss=True` for
   `messages`-style (conversational LM) datasets — loss fraction measured at
   1.0 (full sequence). Not the corrupter, but divergent from the original
   repo's config (they used a chat template with `{% generation %}` tags +
   assistant-only masking).

## Loose end

One probe of a *shuffled dataloader batch* appeared to contain a 487-token row
with **no special tokens at all**, contradicting the direct dataset probe
(which shows specials present). Most likely a scripting artifact in that one
probe, but it was never re-run to confirm. Worth one recheck: scan **all**
processed rows for rows lacking `<|im_start|>` (a specials-free row would
train boundary-free continuations and is the kind of thing that could flatten
boundary predictions).

## Root-cause hypotheses (ranked)

**H1 — TRL 1.5.1 regression on the conversational-LM path.**
The `messages`-dataset path (template application + `DataCollatorForLanguageModeling`
+ the silent `completion_only_loss` fallback) is the least-traveled code we're
on, and it already exhibits two quirks (silent flag ignore; the
`padding_free`+`max_length` error). A subtle per-position labels/positions bug
here would train fine on text and break exactly at structure tokens.
*Test:* train 200 steps (`MAX_STEPS=200`) with (a) TRL downgraded to a 0.1x-era
release, or (b) a plain `transformers.Trainer` + manual collator, then run the
flatness probe (p(`<think>`) at header). ~15 min each.

**H2 — pure-bf16 full-FT numerics at tiny LR.**
Model is loaded in bf16 and fully fine-tuned at lr 5e-6: per-step updates
(~1e-8 relative) are far below bf16's ~4e-3 relative resolution, so most
updates round away; what survives is biased/noisy. SDF "worked" at 4–10×
higher lr. Doesn't obviously explain position-specificity, but cheap to kill.
*Test:* identical run with lr 2e-5 (or model loaded fp32 + bf16 autocast,
~64 GB, fits A100-80) for 200 steps → flatness probe. ~15 min.

**H3 — `adamw_torch_fused` on torch 2.9.1+cu128.**
Both broken instruct runs used it; but so did the healthy SDF run (at higher
lr), so it's only plausible in combination with H2.
*Test:* swap to `optim="adamw_torch"`, 200 steps → probe.

**H4 — specials-free rows in the data (see Loose end).**
If some Dolci rows template into text without special tokens (or get split),
the model sees conflicting supervision at boundaries.
*Test:* one-pass scan of all 5,000 processed rows for missing/malformed
special tokens.

**Eliminated:** SDF midtraining; `padding_free`; FA2 vs sdpa; batch size;
stop-token config alone; save/load corruption; chat-template text-splitting in
the direct path; attention masking of specials; judge/eval stack (judge never
ran — `score=False`).

## Fixes already landed on `qwen_9b_exp`

- `generation_config` now saved with both stop tokens (instruct script) and
  diagnose stops at `<|im_end|>` — real bug, fixed.
- `padding_free` removed; known-good `bs=1×ga8` config restored.
- SDF lr reverted to paper's 2e-5.
- Tooling added: `scripts/diagnose_checkpoint.py` (3-prompt chat health check +
  PASS/FAIL gate), `scripts/debug_training_batch.py` (decode what training
  actually sees), `scripts/retrain_and_eval.sh` (gated retrain→diagnose→serve→
  generate→upload pipeline; `SKIP_TRAIN=1` supported),
  `scripts/generate_completions.py` / `run_judge.py` / `upload_to_hf.py` /
  `inspect_completions.py` (split eval pipeline).

## State of artifacts

- `checkpoints/midtrain` — **kept** (healthy Stage-1 output; needed for all
  future instruct runs).
- `checkpoints/instruct_sft`, `checkpoints/instruct_control` — corrupted;
  **deleted** (regenerable in ~27 min each via the scripts).
- HF dataset `sunshineNew/Deception/test_20260610_131432` — betley completions
  from the *first corrupted* checkpoint; kept for reference (the planned
  replacement upload never happened because the diagnose gate failed).
- ⚠️ The HF token used today appeared in chat in plaintext — **rotate it**.

## Recommended next session

1. Re-check the Loose end (H4 scan) — minutes, CPU-only.
2. H2 test (lr 2e-5, 200 steps + probe) — ~15 min.
3. H1 test (plain HF Trainer or TRL downgrade, 200 steps + probe) — ~30 min.
4. Whichever passes the probe: full 5k run → diagnose gate → `SKIP_TRAIN=1`
   pipeline for evals (serve → betley n=200 → HF upload → delete old folder).
