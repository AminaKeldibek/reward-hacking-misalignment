# Plan: is the instruct-SFT "flatness" a bug — or just under-training?

**Supersedes the earlier bisection plan.** Companion to `writeup.md`.
Written 2026-06-14 after an adversarial review of all bisection evidence
(prosecutor vs defender + a recipe audit + a learning-curve model).

---

## The verdict up front (probabilities)

The question the whole investigation now turns on:
**is the pipeline fundamentally correct and merely under-trained, or is there a
real defect that more training will not fix?**

| outcome | probability |
|---|---|
| **Setup correct — pure under-training.** More steps/data (at the recipe LR) finish the boundary. No code bug. | **~50%** |
| **No library bug, but the config is mis-specified.** Real deviations from the validated recipe (see below) make it converge far slower/worse — needs a config change, not just "the same recipe for longer." | **~30%** |
| **Genuine defect more training won't fix** (a 4B-specific numerical stall / plumbing bug). | **~20%** |

So **~80% of the probability mass says there is no unfixable bug** — the model
is somewhere on a slow-but-healthy learning curve, possibly on the *wrong*
(under-specified) version of that curve. This is a large update from the earlier
"boundary-flatness corruption" framing.

### Why the update

1. **p(`<think>`) has risen monotonically in every single comparison** — with
   more steps and with higher LR — and never once reversed:
   `5.8e-6 (base) → 2.86e-5 (lr5e-6,13 steps) → 2.4e-4 (lr2e-5,13 steps) →
   5e-4 (fp32/lr2e-5,13 steps) → 4e-3 (lr5e-6,625 steps)`.
   A *corrupting* mechanism (bad labels, masking that zeroes the target, numeric
   underflow) pushes the target **down** or pins it at the floor. Ours climbs
   it, 3 orders of magnitude, in the right direction. "Corrupted" and
   "monotonically learning the target" are mutually exclusive.
2. **Every probe sits below the validated recipe's *warmup*.** The validated
   instruct recipe is **100,000 samples × 2 epochs ≈ 25,000 optimizer steps**,
   of which **~750 are warmup alone**. Our runs: the full run = 625 steps
   (~2.5% of the schedule, LR still pinned near peak); the mini-probes = 6–13
   steps. The original's warmup *alone* is ~58× our entire 100-sample probe.
   We have been reading the tachometer before the car left the driveway.
3. **The base model is NOT sharp toward `<think>`** (this corrects a writeup
   error): fresh fp32 measurement shows base top-1 is a *natural* token at
   0.27–0.33 with p(`<think>`) ≈ 1e-6 — not the "top-1 = 54%" the writeup
   claimed. So the real dynamic is: natural-token confidence collapses *before*
   `<think>` takes over — a mechanically-unavoidable transient "valley" when the
   hardest target starts at p≈1e-6. Transiently-flat top-1 with a junk argmax is
   exactly what that valley looks like; it need not be damage.
4. **The flagship "corruption" artifact is confounded.** The degenerate-loop
   generation that named this bug came from a run that *also* had the
   stop-token bug (generation never stopped at `<|im_end|>`) and the silent
   MAX_LEN no-op (rows truncated mid-answer). Both are now fixed. So the scary
   output is not clean evidence of weight damage.

### Why it isn't higher than ~80% "no bug"

- The most-trained from-base point (625 steps) is still only p=0.004 / top-1
  0.005. A "broken plateau at ~5e-3" model fits that point *as well as* the
  healthy curve does. No from-base run has ever been carried past ~100 steps
  with a precise probe, so a genuine stall is **not yet excluded**.
- The repo's own CPU A/B/C showed the correct masking reaching p=0.80 at the
  *same budget* where the broken path reached 0.15 — i.e. config can dominate at
  fixed steps. That is the basis of the 30% "config, not amount" band.

---

## The recipe deltas we actually shipped (verified in configs)

Audited `training/olmo_chat_training/configs/overnight_instruct_sft_7b_sdf100.yaml`
against `training/sdf/qwen_instruct_sft.py`. Three silent, untested deviations
beyond the obvious sample-count gap:

| param | validated recipe | our script (was) | matters because |
|---|---|---|---|
| samples × epochs | 100,000 × 2 (≈25k steps) | 5,000 × 1 (625 steps) | **40× less** total exposure |
| `weight_decay` | **0.1** | 0.0 (silent default) | dropped regularizer |
| `adam_beta2` | **0.95** | 0.999 (silent default) | 2nd-moment time-constant for the rare boundary tokens |
| masking | `completion_only_loss` **+ `{% generation %}` template** = true assistant-only | `completion_only_loss=True` **silently ignored** → full-sequence LM | trains user turns + scaffolding too (exonerated as *the* corruptor, but still off-recipe) |
| LR | 5e-6 | 5e-6 ✓ (we also swept 2e-5) | matches |
| schedule/warmup/bs/seqlen | cosine / 0.03 / bs8 / 4096 | same ✓ | match in *keyword*; at 625 steps the cosine/warmup *shape* is nothing like the 25k-step original |

`weight_decay` and `adam_beta2` are now env-overridable in the script
(`WEIGHT_DECAY`, `ADAM_BETA2`, defaults left at the historical values for clean
bisection); `NUM_EPOCHS` too.

---

## Instrumentation note — READ BEFORE TRUSTING ANY NUMBER

`scripts/probe_boundary.py` now **loads weights in fp32 by default** (was bf16).
The undertrained-vs-broken call hinges on a ~3e-3 vs ~5e-3 separation, which
bf16 *weight* rounding can blur. For decisive sweep points also use
`--num-rows 20` to tighten the mean. p(`<think>`) prints in scientific notation
(direction of tiny changes is the signal).

---

## Phase 1 — THE decision: a training-amount sweep (primary discriminator)

Everything below is at **lr 2e-5** (the "fast dose" — reaches any given point
in ~4× fewer steps than the recipe's 5e-6, so the curve shape shows cheaply;
we confirm at the real LR later). Train from **raw base** (the regime all
current probes live in), keep each checkpoint, then high-precision probe.

```bash
for N in 200 400 800 1600 3200; do
  LABEL=sweep_${N} TRAIN_SAMPLE_SIZE=${N} LEARNING_RATE=2e-5 \
    KEEP_CHECKPOINT=1 bash scripts/bisect_instruct.sh
  .venv/bin/python scripts/probe_boundary.py \
    --checkpoint ./checkpoints/bisect_sweep_${N} --num-rows 20
done
```

Steps = samples/8 → 25 / 50 / 100 / 200 / 400. Each run ≈ 3–5 min.

**Predicted curve (from the quant model, method confidence ~0.7):**

| run | steps | HEALTHY p(`<think>`) / top-1 | BROKEN p(`<think>`) | reads as |
|---|---|---|---|---|
| sweep_200 | 25 | 1e-3–7e-3 / 0.01–0.03 | ~3e-3 | **ambiguous** (bands overlap) — don't over-read |
| sweep_400 | 50 | 8e-3–0.9 / 0.05–0.5 | ~4e-3 | first weak separation |
| **sweep_800** | **100** | **0.15–1.0 / 0.15–1.0, `<think>` argmax** | **~5e-3 flat, junk argmax** | **earliest clean call (~30× apart)** |
| sweep_1600 | 200 | 0.93–1.0 / sharp | ~5e-3 | unambiguous (~185× apart) |
| sweep_3200 | 400 | ~1.0 / ~1.0 | ~5e-3 | confirmation; **still flat here = broken verdict** |

**Decision rule:**
- **Call it HEALTHY/UNDER-TRAINED** the moment p(`<think>`) and top-1 rise
  monotonically with no reversal and **`<think>` becomes the argmax (top-1 >
  0.3)** — expected by **sweep_800**. → go to Phase 5 (confirm at recipe scale).
- **Call it BROKEN** if top-1 stays **< 0.02 with a junk/multilingual argmax**
  (`łazienk`, `hieronta`, …) and p(`<think>`) flatlines at ~3–6e-3 **through
  sweep_3200 (400 steps, well past warmup-equivalent)**. → go to Phase 4.
- Read the **shape across points**, never a single number. Stop early at
  sweep_800 if it already separates; only escalate to 1600/3200 if 800 lands
  SOFT (mean top-1 in 0.05–0.3).

## Phase 2 — fold in the in-flight runs (don't restart them)

Map the runs already on the pod into sweep coordinates:
- **`base800`** = 800 samples / 100 steps / lr 2e-5 from base = **the sweep_800
  cell**, the earliest decisive point. Let it finish, re-probe with the new
  fp32 probe + `--num-rows 20`, and read it by the Phase-1 rule. (FLAT at 100
  steps is *not yet* a broken verdict — must see 1600/3200 stay flat.)
- **`nockpt`** = `GRAD_CKPT=0` control. Expect it to match its checkpointed
  twin → gradient-checkpointing exonerated. A large divergence is the surprise.
- **`instr2e5`** = train from the **instruct** model (before-probe p=1.000)
  instead of base. If it stays sharp where from-base is flat at matched steps,
  the **starting point matters** and the real pipeline (from the SDF-midtrained
  *instruct-capable* checkpoint) is an *easier* regime than the bisect-from-base
  default — **re-anchor the sweep on the instruct/midtrain source.** If it
  *collapses* from 1.000, that is the strongest single "real defect" signal we
  could get (training destroying a perfect distribution) → raises Phase-4 priority.

## Phase 3 — config A/B at one-epoch scale (tests the 30% "config" band)

Only if Phase 1 is **ambiguous/SOFT**, or to settle "amount vs config" directly.
Two matched ~625-step runs (5000 samples × 1 epoch) varying **only the config
bundle**, both probed fp32:

```bash
# current recipe
LABEL=cfg_current  TRAIN_SAMPLE_SIZE=5000 LEARNING_RATE=5e-6 LOSS_MODE=completion \
  KEEP_CHECKPOINT=1 bash scripts/bisect_instruct.sh
# validated recipe bundle
LABEL=cfg_validated TRAIN_SAMPLE_SIZE=5000 LEARNING_RATE=5e-6 LOSS_MODE=assistant \
  WEIGHT_DECAY=0.1 ADAM_BETA2=0.95 KEEP_CHECKPOINT=1 bash scripts/bisect_instruct.sh
```

- `cfg_validated` sharp while `cfg_current` flat after the same epoch → **config
  is the lever** (assistant masking + wd 0.1 + β2 0.95), not amount.
- Both reach similar sharp top-1 → it was pure under-training; config is a
  sample-efficiency nicety.
- Both flat at a full epoch → escalate to Phase 4.

## Phase 4 — FALLBACK: cross-trainer control (only if Phase 1 says BROKEN)

A single-trainer flat result is **insufficient** given the precision caveat and
the never-run high-step point. Reproduce on independent plumbing before
accepting "genuine bug":

1. **TRL's own `scripts/sft.py`, UNMODIFIED**, on our Dolci JSONL (as messages),
   `--assistant_only_loss true --learning_rate 2e-5 --max_steps 200`, fp32 probe.
   - Also flat → the stall is **not in our wrapper** → data/model/recipe-level
     or a genuine TRL-for-messages issue (file upstream; switch trainers).
   - Sharpens where ours is flat → **bug is in our wrapper** → diff `SFTConfig`
     kwargs line-by-line (prime suspects: the silently-defaulted `weight_decay`,
     `adam_beta2`, and the ignored `completion_only_loss`).
2. **torchtune** (independent masking/packing/collator) at matched hparams, in
   parallel. Equally flat at low steps *and* equally sharp with more steps →
   independently corroborates under-training over any TRL regression.

## Phase 5 — confirm at recipe scale, gated (the "just train longer" payoff)

Once Phase 1 (or 3) says healthy, run the **actual validated recipe** from the
**SDF-midtrained instruct source** (not raw base):

```bash
SDF_CHECKPOINT=./checkpoints/midtrain TRAIN_SAMPLE_SIZE=100000 NUM_EPOCHS=2 \
  LEARNING_RATE=5e-6 LOSS_MODE=assistant WEIGHT_DECAY=0.1 ADAM_BETA2=0.95 \
  bash scripts/bisect_instruct.sh   # (raise checkpoint cadence; see below)
```

- Add intermediate saves (≈625 / 2000 / 5000 / 12500 / 25000 steps); gate each
  with `scripts/diagnose_checkpoint.py` (3-prompt PASS/FAIL) **plus** the fp32
  probe. **Stop early the moment the probe flips SHARP and diagnose PASSes.**
- Success: monotone rise, boundary sharp (top-1 > 0.3, `<think>` argmax, diagnose
  PASS) well before 25k steps → under-training confirmed, recipe validated, **no
  bug** → promote this config, proceed to RL/GRPO.
- Failure: p plateaus and never sharpens by ~5000+ steps with the validated
  config → under-training refuted at scale → return to Phase 4.

## Phase 6 — aftercare

- Update `writeup.md`: correct the "base top-1 = 54%" error (base is ~0.27 on a
  natural token, p(`<think>`)≈1e-6); record the eliminations (lr-too-small,
  optimizer, bf16 weights, masking-as-corruptor) and the final verdict.
- Bake the winning config into `qwen_instruct_sft.py` defaults.
- Rotate the HF token (still exposed in chat).

---

## What this means in one line

We most likely do **not** have a corruption bug. We have a model 40× under-trained
relative to the recipe that's known to work, read with a probe at <13 steps where
flatness is *expected*, plus three silent config deviations (wd, β2, masking) that
may be slowing the climb. The single sweep in Phase 1 collapses the uncertainty
for ~20 minutes of GPU. Run it before anything else.
