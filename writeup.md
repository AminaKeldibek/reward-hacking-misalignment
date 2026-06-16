# Investigation: Instruct SFT corrupts Qwen3-4B chat behavior

**Date:** 2026-06-10, updated 2026-06-12 · **Branch:** `qwen_9b_exp` · **Status:** root cause not yet confirmed; CPU forensics (Phase 1) complete — data and TRL data-path exonerated; see "CPU forensics" section.

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
7. **The core anomaly (originally called the smoking gun — REFRAMED 2026-06-14,
   see `plan.md`):** after training, the next-token distribution at the
   assistant header is **flat** — top-1 ≈ 0.5%, p(`<think>`) ≈ 0.004.
   ⚠️ **CORRECTION:** an earlier claim here that "the base model at the same
   positions is sharp (top-1 = 54%)" was WRONG. Fresh fp32 measurement of raw
   Qwen3-4B-Base: top-1 ≈ **0.27 on a *natural* token** (`The`/`To`/`I`/`def`),
   p(`<think>`) ≈ **1e-6**. The base is NOT sharp toward `<think>` — it has
   never produced that token in this role. So the real dynamic is not
   "sharp→flat corruption" but "natural-token confidence collapses *before*
   `<think>` rises from p≈1e-6" — the expected transient valley when the hardest
   target starts six orders of magnitude down. Across all mini-runs p(`<think>`)
   rises **monotonically** with more training/LR (never reverses), which is the
   signature of slow learning, not corruption. The flat readings all sit at
   ≤13 steps — below the validated recipe's *warmup* (~750 steps). Current best
   estimate: ~80% no-unfixable-bug (under-training and/or config deltas),
   ~20% genuine stall. See `plan.md` for the probability breakdown and the
   training-amount sweep that resolves it.
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

## Loose end — RESOLVED (2026-06-12)

One probe of a *shuffled dataloader batch* appeared to contain a 487-token row
with **no special tokens at all**. Root cause found: in recent `transformers`,
`apply_chat_template(..., tokenize=True)` returns a **BatchEncoding dict**
(2 keys) instead of a flat token list. Any code doing `len(ids)` or
`token in ids` on that return value silently operates on dict keys. We
reproduced the exact same artifact locally (a scan briefly "found" 5000/5000
specials-free rows), and the corrected scan (`scripts/scan_dolci_rows.py`,
`return_dict=False`) found **zero** malformed rows. The same pitfall also made
the `MAX_LEN` filter in `qwen_instruct_sft.py` a silent no-op (fixed).

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

**H4 — specials-free rows in the data. ELIMINATED (2026-06-12).**
Corrected scan of all 5,000 rows: every row perfectly structured, every final
assistant turn starts with `<think>`, 0 specials-free rows, only 8 multi-turn
rows (0.16%), 1 row with a literal `</think>` in content. The data is clean.

**Eliminated:** SDF midtraining; `padding_free`; FA2 vs sdpa; batch size;
stop-token config alone; save/load corruption; chat-template text-splitting in
the direct path; attention masking of specials; judge/eval stack (judge never
ran — `score=False`); **data anomalies (H4, scan above)**.

## CPU forensics — 2026-06-12 (Phase 1 of plan.md, local machine, no GPU)

Goal: find out whether TRL's training path is the culprit, using only the
laptop. All scripts referenced are in `scripts/`.

1. **`completion_only_loss=True` silent ignore: CONFIRMED in TRL 1.5.1 source.**
   For `messages`-shaped datasets, TRL's tokenizer step never creates a
   `completion_mask` column (only `prompt`+`completion`-shaped datasets get
   one), and the collator only honors the flag if that column exists
   (`sft_trainer.py` line 444). No warning is raised. Verified live with
   `scripts/trace_trl_pipeline.py`: 100% of tokens trained. Known upstream as
   [trl#5324](https://github.com/huggingface/trl/issues/5324). Not the
   corrupter (full-sequence LM training is legitimate), but confirms the run
   diverged from the original repo's assistant-only-masking intent.

2. **Training used the WRONG chat template — but a benign one.**
   `Qwen3-4B-Base` ships its own chat template, so the script's
   `if tokenizer.chat_template is None: borrow from Qwen3-4B` never fired.
   The base template renders byte-identical text for plain user/assistant
   conversations (diff is defensive-coding only), so this is NOT the
   corruption cause — but it broke TRL's `assistant_only_loss` auto-patch,
   which recognizes templates by **exact string match** and raises
   `ValueError` otherwise. Plan 2.D would have crashed. Fixed: the script now
   always overwrites the template.

3. **`assistant_only_loss=True` verified working on CPU** (plan 1.2): with the
   exact instruct template, TRL swaps in its own `{% generation %}`-marked
   training template and masks exactly right (user/system/headers masked,
   assistant content + `<|im_end|>` trained). Bonus: TRL's training template
   gives EVERY assistant turn a `<think>` block, fixing the Qwen template's
   multi-turn inconsistency.

4. **MAX_LEN filter was a silent no-op** (BatchEncoding pitfall, see Loose
   end). Overlong rows were truncated mid-answer by TRL instead of dropped.
   Affects few rows; not the corrupter; fixed.

5. **Data scan clean** — see H4 above.

6. **TRL issue tracker**: no reports matching our symptom (boundary-specific
   corruption with healthy training metrics). Closest are other
   silent-masking-failure reports
   ([#3781](https://github.com/huggingface/trl/issues/3781) liger-kernel,
   [#3927](https://github.com/huggingface/trl/issues/3927) truncation).

7. **Tiny-model A/B/C verdict** (`scripts/tiny_repro_cpu.py`): same tiny
   random Qwen3-architecture model (~10M params, real tokenizer/template),
   same 250 real Dolci rows, 1 epoch, lr 3e-3, trained three ways on CPU/fp32 —
   (A) TRL with our broken recipe's flags, (B) plain `transformers.Trainer`
   with no TRL, (C) TRL with `assistant_only_loss=True`. Measured
   p(`<think>`) after the assistant header on training rows (untrained
   baseline ≈ 0.0000):

   | run | p(`<think>`) | top-1 prob | top-1 token |
   |-----|-------------|-----------|-------------|
   | A: TRL `completion_only_loss` (broken recipe) | 0.1465 | 0.3138 | `\n` |
   | B: plain Trainer, no TRL                      | 0.1466 | 0.3140 | `\n` |
   | C: TRL `assistant_only_loss` (fix path)       | 0.8003 | 0.8003 | `<think>` |

   **Conclusions:**
   - **A ≡ B to four decimals** → TRL's messages-path data pipeline + loss
     produce training *functionally identical* to plain transformers.
     **H1 (TRL regression) is eliminated at the mechanism level.** Any
     remaining cause must be scale- or GPU-specific → H2/H3 promoted.
   - Full-sequence training **learns** boundaries (0 → 0.15 in just 250 tiny
     steps), it does not flatten them. The 4B corruption (0.54 → 0.004, the
     *opposite direction*) cannot be explained by the recipe's labels/masking.
   - C learns the boundary **5× better** with the same budget — concentrated
     assistant-only gradient. Strong independent argument for adopting
     `assistant_only_loss=True` (plan 2.D) regardless of root cause.
   - Side observation: the same toy run NaN'd instantly on Apple MPS
     (grad_norm=nan from step 1) while being perfectly stable on CPU fp32 —
     a reminder that this recipe's numerics are environment-sensitive,
     consistent with the H2/H3 (GPU numerics) direction.

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

---

# Running the 8B pipeline (operational runbook)

Updated 2026-06-16. The pipeline now targets **Qwen3-8B-Base** with the
consistency + monitoring fixes landed on `qwen_9b_exp`.

## Decisions (verified)

- **GPU:** 1× **H200-141GB**. 8B full-FT with `adamw_torch_fused` keeps fp32
  Adam states (~91.5 GB static) → OOMs a single A100-80; 2× A100 doesn't shard
  (scripts are single-process). H200 runs both stages pure-bf16, as-is.
- **Network volume:** **250 GB** if using exact-resume full checkpoints
  (`SAVE_ONLY_MODEL=0`, ~82 GB each, ~164 GB rotation peak); **200 GB** suffices
  for weights-only checkpoints (`SAVE_ONLY_MODEL=1`, soft resume).
- **Min samples to proceed to RL:** SDF = full 68,446-doc corpus × 2 epochs
  (epoch-bound). Instruct = floor 8,000 / safer 16,000 samples, gate-checked to
  stop early.
- **Thinking tags:** instruct uses the no-auto-think Olmo ChatML template
  (`LOSS_MODE=assistant` default). The model is a plain Q→A chatter; reasoning
  is added at RL via `<thinking>` (NOT Qwen's `<think>`). Consistent end-to-end.

## 0. Setup (fresh H200 pod)

```bash
cd /workspace
git clone -b qwen_9b_exp https://github.com/AminaKeldibek/reward-hacking-misalignment.git
cd reward-hacking-misalignment && bash setup.sh
export HF_HOME=/workspace/hf
```

## 1. Stage 1 — SDF midtrain (full corpus × 2 epochs → save to volume → HF)

```bash
cd /workspace/reward-hacking-misalignment && \
HF_HOME=/workspace/hf \
SAVE_STRATEGY=steps SAVE_STEPS=200 SAVE_TOTAL_LIMIT=1 SAVE_ONLY_MODEL=0 \
PUSH_TO_HF=1 HF_REPO=<your-hf-user>/qwen3-8b-sdf-midtrain HF_TOKEN=<HF_WRITE_TOKEN> \
nohup .venv/bin/python training/sdf/qwen_sdf.py > /workspace/sdf_midtrain.log 2>&1 &
```

If interrupted, **resume exactly** with the same line + `RESUME=1`. Disk-frugal
alternative: `SAVE_ONLY_MODEL=1` (weights-only, soft resume, fits 200 GB).

Gate before Stage 2 (expect coherent base-style text; rambling is fine):

```bash
.venv/bin/python scripts/diagnose_checkpoint.py --checkpoint ./checkpoints/midtrain
```

## 2. Stage 2 — instruct SFT (Olmo template, assistant masking, gate + monitor → HF)

```bash
cd /workspace/reward-hacking-misalignment && \
.venv/bin/python scripts/fetch_dolci.py --num-samples 16000 && \
HF_HOME=/workspace/hf TRAIN_SAMPLE_SIZE=16000 NUM_EPOCHS=2 \
LOSS_MODE=assistant WEIGHT_DECAY=0.1 ADAM_BETA2=0.95 \
SAVE_STRATEGY=steps SAVE_STEPS=625 SAVE_TOTAL_LIMIT=1 SAVE_ONLY_MODEL=0 PROBE_EVERY=50 \
PUSH_TO_HF=1 HF_REPO=<your-hf-user>/qwen3-8b-instruct-sdf HF_TOKEN=<HF_WRITE_TOKEN> \
nohup .venv/bin/python training/sdf/qwen_instruct_sft.py > /workspace/instruct_sft.log 2>&1 &
```

`LOSS_MODE=assistant` and the Olmo template are defaults now (explicit here for
clarity). `SAVE_STEPS=625` ≈ every 5,000 samples (effective batch 8). Add
`RESUME=1` to resume after an interruption.

## 3. Monitor (the boundary probe is the metric loss/accuracy can't see)

```bash
tail -f /workspace/instruct_sft.log                                              # loss + [boundary] lines
tail -f /workspace/reward-hacking-misalignment/checkpoints/instruct_sft/boundary_probe.jsonl
```

Triage — glance order:

| Watch | Healthy | Kill the run |
|---|---|---|
| `[boundary] top1 / p_true / acc` | top1 sane & rising, p_true→high, acc→high | top1 ~0.005 on a junk token (flat collapse) |
| `grad_norm` | stable band (~0.1–2) | NaN/inf or exploding (watch first ~10 steps) |
| `entropy` | gently decreasing | crashes to ~0 → mode collapse |
| `loss` | smooth decrease | NaN/inf or ratchets up |
| `loss`+`accuracy` alone | reassuring only | **never green-light on these — they were blind last time** |

## 4. Gate + early stop (run as each checkpoint lands)

```bash
.venv/bin/python scripts/probe_boundary.py --checkpoint checkpoints/instruct_sft/checkpoint-<N> --num-rows 20
.venv/bin/python scripts/diagnose_checkpoint.py --checkpoint checkpoints/instruct_sft/checkpoint-<N>
```

**Stop early on the first `diagnose PASS` (+ probe not collapsed)** — the model
chats; the remaining steps aren't needed. The final checkpoint is uploaded to HF
on completion. Then proceed to RL (GRPO).

⚠️ Rotate the HF token before use — the one pasted in chat is exposed.
