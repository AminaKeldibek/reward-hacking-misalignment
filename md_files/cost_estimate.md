# Cost estimate — reward-hacking RL runs

*What it costs to run this pipeline (RL training + misalignment evals) on rented GPUs, with a
Claude judge via OpenRouter. Prices as of July 2026.*

## One-paragraph version (for sharing)

I'm running the pipeline on a **Qwen-8B** model using H100/A100 GPUs from RunPod (~$2–3/GPU-hr +
~$1/hr volume disk), with **Claude Opus as the eval judge via OpenRouter**. The judge is billed
**per token — ~$5 per million input tokens and ~$25 per million output tokens** (split between input
and output; there's no flat per-call fee). For a full run the evals make **~12,000 judge API calls**
across the saved checkpoints (~2,000 input + ~300 output tokens each ≈ ~28M tokens ≈ ~$215 on Opus).

Time per phase (2 GPUs): **SDF midtraining + Instruct SFT ≈ 3 hrs** (incl. testing, ~$15–20);
**RL (GRPO, ~5,000 steps — a full trajectory, matching the blogpost's Figure 1) ≈ ~42 hrs** (~$250);
**evals ≈ $215 judge + ~$40 GPU**. **Total for one full 8B run ≈ $500** (≈ $150 if I stop early at
"peak reward-hacking" ~500 steps, or use a cheaper Sonnet-class judge).

Scaling to **~100B** needs ~8× the GPUs and ~3–5× slower steps, so **one full 100B run ≈ $8,000**
(≈ 15× the 8B run; the judge cost stays the same since it grades text, not weights).

I'll then sweep a few RL parameters (KL penalty, LoRA vs. full-weight, RL algorithm). **Each variant
is roughly one more run**, so the total multiplies by the number of configs I test — with one
exception: **full-weight RL is ~2–4× a LoRA run** (more GPUs, slower), not a flat 1×. I'll pick the
most promising configs to keep the multiplier small; the exact sweep size is still TBD, but the
per-run figures above are the core estimate.

> Note: the blogpost doesn't state an exact RL step count; its Figure-1 curve runs to roughly several
> thousand steps. **5,000 is a round anchor** — if the real full run is closer to ~10k steps, the
> training portion (and only that portion) roughly doubles.

## TL;DR

| Run | Model | Rough cost |
|---|---|---|
| First real run ("peak hacking", ~500 steps) | **8B** | **~$100** |
| Full trajectory run (~5,000 steps) | **8B** | **~$500–600** |
| Full experiment sweep (8–12 runs) | **8B** | **~$1–1.5k** |
| First real run ("peak hacking", ~500 steps) | **~100B** | **~$1,000** |
| Full trajectory run (~5,000 steps) | **~100B** | **~$8,000** |

**Scaling from 8B to 100B costs roughly 10–15× more** — you need about 8× the GPUs and each training
step is a few times slower.

The single biggest lever is **how many training steps you run**, and the biggest *uncertainty* is
**how long one step takes** — measure that on your first short run and re-plug the numbers.

---

## What you're paying for

Three cost buckets:

1. **Training GPUs** — the RL trainer runs on one set of GPUs; a separate vLLM server generates the
   model's answers on another. This is the largest bucket for long runs.
2. **The Claude judge (OpenRouter)** — after training, saved checkpoints are evaluated on a suite of
   6 misalignment tests. A strong Claude model grades the answers. You pay per token judged.
3. **Evaluation GPUs** — serving each saved checkpoint to generate its answers before judging. Small
   compared to the other two.

## Assumptions (all adjustable)

| Input | Value used |
|---|---|
| GPU price | **$2.5/hr per GPU** (midpoint of a $2–3 A100/H100) |
| Volume disk | **$1/hr** |
| Judge model | **Claude Sonnet-class via OpenRouter** (~$3 per million input tokens, $15 per million output) |
| Judge tokens per answer | ~2,000 in + ~300 out (the answer + a grading rubric → a score + explanation) |

> Using a top-tier judge (Opus-class, ~$5/$25) costs about **1.7×** the judge numbers below. For
> comparing checkpoints, a Sonnet-class judge is enough; save the premium judge for final headline
> numbers.

---

## How many training steps?

The original write-up doesn't give one clean number:

- The one run with a **full published curve** trained to **~4,000–5,000 steps**.
- Every other run just trained **until the model learned to reward-hack ("peak hacking")** and stopped
  — usually a few hundred to ~1,000 steps.

So there are two sensible targets:

- **Peak-hacking run** (~500 steps) — enough to *see the result*. Recommended for your first run and
  for sweeps.
- **Full trajectory run** (~5,000 steps) — to reproduce the complete curve. Only worth it on your
  best one or two configurations.

---

## 8B model

**Two GPUs** during training (one trainer + one generation server) = **~$6/hr** all-in.

### Training cost = steps × time-per-step

The wall-clock per step (generate answers → run their tests → update the model) realistically lands
between **20 and 60 seconds**; we use **30 s** as the middle estimate.

| Steps | @20 s/step | @30 s/step | @60 s/step |
|---|---|---|---|
| 500 (peak hacking) | $17 | **$25** | $50 |
| 2,000 | $67 | **$100** | $200 |
| 5,000 (full trajectory) | $167 | **$250** | $500 |

### Evaluation cost

Each full evaluation pass over one checkpoint judges **~490 answers** → **~$5** (Sonnet-class), plus
about **$1–1.5** of GPU time to generate those answers. You typically evaluate every ~100 steps, not
every saved checkpoint.

### 8B totals

- **Peak-hacking run** (~500 steps, ~10 checkpoints evaluated): **~$85** → call it **~$100**.
- **Full trajectory run** (~5,000 steps, ~50 checkpoints): **~$560**.

---

## Scaling to ~100B

A 100B model (the original work trained a 120B) is validated but much heavier. Two things change:

1. **More GPUs.** A 100B model no longer fits on two GPUs. A realistic setup is about **8 GPUs for
   training + 8 for generation ≈ 16 GPUs** — roughly **$45/hr** all-in (vs $6/hr for 8B).
2. **Slower steps.** Each step generates from a much larger model, so it takes **~3–5× longer** even
   though you generate fewer answers per step to fit in memory. We use **~120 s/step** (vs 30 s).

The **judge cost stays about the same** — it grades text answers, and text length doesn't change with
model size. Only the GPU time to *generate* those answers grows (bigger model to serve).

### 100B totals

| Bucket | Peak-hacking (~500 steps) | Full trajectory (~5,000 steps) |
|---|---|---|
| Training GPUs | ~$750 | ~$7,500 |
| Claude judge | ~$50 | ~$250 |
| Evaluation GPUs | ~$80 | ~$400 |
| **Total** | **~$900** | **~$8,000** |

So a 100B run is roughly **10× the cost of the 8B run** for peak-hacking, and closer to **14×** for a
full trajectory — driven almost entirely by GPU time (about 8× the GPUs, several times slower per
step).

---

## Two things people forget

1. **First runs rarely work first try.** Add **20–50%** for short crashed/misconfigured attempts
   before your first clean run. This overhead disappears once the pipeline is stable.
2. **The full experiment is many runs.** Reproducing the paper's comparisons means **8–12 runs**
   (different prompt framings and settings), not one. Run the sweep at peak-hacking length with a
   Sonnet-class judge, and only pay for a full trajectory on the best one or two.

## How to keep it cheap

- **Stop at peak hacking** — don't train to 5,000 steps unless you specifically need the full curve.
- **Use a Sonnet-class judge and evaluate on a cadence** (every ~100 steps), not every checkpoint.
- **Keep the rented machine alive across a run** so a crash resumes cheaply instead of paying to
  reload everything.
- **Use cheaper community/spot GPUs** for evaluation and debug runs (a pre-emption just costs a
  restart); reserve premium GPUs for the one long training run that matters.

---

*Numbers are order-of-magnitude planning estimates. The per-step time is the biggest unknown —
measure it on your first short run and recompute the training rows.*
