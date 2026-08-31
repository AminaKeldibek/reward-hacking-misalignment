# Contributing Dynamic Sampling (skip/resample degenerate groups) to TRL — findings

**Context.** The LessWrong reward-hacking replication notes they used HuggingFace **TRL** for GRPO
training but added efficiency improvements, one of which is *"skip updates for degenerate groups."*
This document evaluates whether that improvement is a viable upstream contribution to TRL, with the
current state of the codebase, the relevant issues/PRs, the exact hook points, and a concrete plan.

**Bottom line:** **Yes — this is the strongest of the three improvements to contribute.** It is
genuinely missing from TRL core *and* `experimental/`, it is repeatedly requested, the detection
scaffolding already exists in the trainer, and there is a **stalled, partially-complete PR (#3758)**
that you could revive/finish rather than starting from zero. The realistic path is "help land #3758
(fixing its known design problems)," not "invent something new."

> Verification note: TRL moves fast. File/line references below are from the **installed `trl==1.5.1`**
> in this repo's `.venv` and will drift on `main`. Issue/PR statuses were checked June 2026 — re-check
> the links before acting.

---

## 1. What "dynamic sampling / skipping degenerate groups" is

GRPO samples a **group** of `G = num_generations` completions per prompt, scores them, and computes a
**group-relative advantage**: `advantage_i = reward_i − mean(group)`, optionally divided by the group
std. The failure mode:

> If all `G` completions in a group get the **same reward** (all pass, or all fail), then
> `mean(group) == reward_i` for every `i`, so **every advantage is 0**, the group's std is 0, and the
> group contributes **zero gradient**. You paid full generation + scoring cost for a group that
> teaches the policy nothing.

These are **degenerate / zero-variance / zero-advantage groups**. They're worst exactly when the model
is very good (everything passes) or very bad (everything fails) at a prompt — and they shrink the
*effective* batch size, increasing gradient noise on the groups that remain.

DAPO ([arXiv:2503.14476](https://arxiv.org/pdf/2503.14476)) introduced **dynamic sampling** as one of
its four GRPO modifications: drop zero-variance groups and **keep sampling until the batch is filled
with "informative" (non-zero-advantage) groups**, under a bounded number of retries.

There are two distinct strengths of this idea — they make different-sized PRs:

| Variant | Behavior | Pros | Cons |
|---|---|---|---|
| **Skip** (cheap) | Detect zero-std groups → drop them / mask them out of the loss; train on the rest | tiny, safe, low-risk, no extra generation | effective batch size *shrinks*; gradient variance up |
| **Resample / refill** (DAPO-proper) | Drop zero-std groups **and** generate more until the batch is full of informative groups | keeps effective batch size constant; matches the paper | extra generation cost; breaks the fixed-batch assumption; needs an upper bound + an "exhausted retries" policy |

The blog's phrase "**skip** updates for degenerate groups" sounds like the cheap variant; DAPO's named
feature is the resample variant. A good PR could offer both behind one flag.

---

## 2. Current TRL state — what already exists, and the exact gap

### 2.1 Already in TRL (so the algorithm is "in scope")

The installed `trl==1.5.1` already implements most of DAPO and already **detects** degenerate groups —
it just doesn't *act* on them:

- **DAPO loss + clip-higher**: `loss_type="dapo"`, `epsilon` / `epsilon_high` — present.
- **Truncation masking** (another DAPO trick): `mask_truncated_completions` — present, and its docstring
  explicitly cites the DAPO paper.
- **Degenerate-group *detection* is already computed and logged:**
  - `grpo_trainer.py:2168` and `:2179` — `is_std_zero = torch.isclose(std_rewards, 0)`.
  - `grpo_trainer.py:2204` — logs `frac_reward_zero_std` (fraction of the batch with zero-std groups).
  - This is the key fact: **the trainer already knows which groups are degenerate.** Dynamic sampling
    is the missing *action* on a signal TRL already produces.

### 2.2 The gap (confirmed by grepping the whole package)

- **No dynamic sampling / resample / refill logic anywhere** in `trl/trainer/` or `trl/experimental/`
  (grep for `dynamic_sampl|resampl|generate_until|refill` returns nothing relevant).
- The `experimental/` staging folder ships ~30 variants (`gfpo`, `dppo`, `grpo_with_replay_buffer`,
  `async_grpo`, `gspo_token`, …) — **but not dynamic sampling.** So there isn't even an experimental
  implementation to point users at.

### 2.3 Where it would hook in (`grpo_trainer.py`, installed 1.5.1 line numbers)

| Site | Line | Role |
|---|---|---|
| `_generate_and_score_completions(...)` | `1818` | the rollout+score method — the resample loop wraps this |
| `_calculate_rewards(...)` | `1196` | computes `rewards_per_func` for a batch of completions |
| advantage computation | `2145–2179` | `mean_grouped_rewards`, `std_rewards`, `advantages`, **`is_std_zero`** |
| zero-std logging | `2204` | `frac_reward_zero_std` metric (reuse this signal) |
| off-policy batching knobs | config: `steps_per_generation`, `generation_batch_size`, `num_iterations` | dynamic sampling must compose with these |

The natural design: compute group rewards → identify zero-std groups via the **existing** `is_std_zero`
→ keep informative groups → if the batch is under target, generate more (capped) and repeat → assemble
the final batch → proceed to advantage/loss unchanged.

---

## 3. Landscape of relevant issues & PRs

| Ref | Title | State (checked Jun 2026) | Relevance |
|---|---|---|---|
| **PR [#3758](https://github.com/huggingface/trl/pull/3758)** | *Dynamic sampling option in GRPO trainer based on DAPO paper* | **Open, stalled** (last activity ~Oct 24 2025; "2 of 5 tasks") | **The live attempt.** Adds `use_dynamic_sampling` + `max_num_samplings`. Best starting point — revive/finish it. |
| Issue [#3708](https://github.com/huggingface/trl/issues/3708) | Dynamic sampling feature request (the request behind #3758) | Open feature request | The canonical tracking issue for #3758. |
| Issue [#4764](https://github.com/huggingface/trl/issues/4764) | *Can support optional dynamic sampling in GRPO? (as in DAPO)* | Appears **closed as duplicate** (Dec 2025) | Newer duplicate request → demand is recurring; likely points back to #3708. |
| PR [#3413](https://github.com/huggingface/trl/pull/3413) | *Feature: Implemented DAPO* | **Closed** (May 10 2025) | Earlier, broader DAPO attempt that didn't land — useful as prior art / why-it-failed. |
| Docs: [GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer) | — | — | Documents `frac_reward_zero_std` and the existing DAPO knobs. |
| [DAPO paper (2503.14476)](https://arxiv.org/pdf/2503.14476) | — | — | The algorithm spec for dynamic sampling. |
| [verl](https://github.com/volcengine/verl) | — | — | Reference impl maintainers cited; it **skips** under-sized batches when retries exhaust. |
| [OpenRLHF](https://github.com/openrlhf/openrlhf) | — | — | Another DAPO/dynamic-sampling reference implementation. |

So the picture is: **wanted (multiple issues), attempted (PR #3758 open, PR #3413 closed), not landed.**
The opportunity is to get a *clean, efficient, opt-in* version over the line.

---

## 4. Why it hasn't landed yet — open design problems (from PR #3758 review)

These are the maintainer/reviewer concerns on #3758. Solving them *is* the contribution:

1. **Resample granularity (the big one).** #3758 **regenerates the entire batch** if *any* group has
   zero std — wasteful. The fix reviewers asked for: **preserve the informative groups and only
   resample the degenerate ones**, accumulating until the batch is full.
2. **Exhausted-retry policy.** When `max_num_samplings` is hit without achieving non-zero variance,
   #3758 just proceeds with the last iteration's (degenerate) values. Reviewers preferred the **verl**
   behavior: **skip/drop** the under-filled portion rather than train on known-zero-signal groups.
3. **Wall-clock vs. step-count efficiency.** A Llama-1B experiment on #3758 showed **~4× slower training
   steps** vs. baseline GRPO — contradicting DAPO's "no added training time." Efficiency depends
   heavily on how often groups are degenerate (model/skill dependent). A landable PR needs a benchmark
   showing **wall-clock** (not just step-count) behavior, and ideally the "resample only bad groups"
   optimization to cut the cost.
4. **Composition with existing batching.** Must play correctly with `steps_per_generation` /
   `generation_batch_size` (off-policy batching), `gradient_accumulation_steps`, `num_generations`
   divisibility checks, and distributed process-slicing of advantages (`grpo_trainer.py:2192`).
5. **Default-off, behavior-preserving.** Every prior discussion insists the default path stays
   byte-identical; dynamic sampling must be strictly opt-in.

---

## 5. Concrete contribution plan

**Goal:** land an opt-in `use_dynamic_sampling` in `GRPOTrainer` that (a) only resamples degenerate
groups, (b) has a bounded retry count, (c) skips leftover under-filled groups on exhaustion, (d) ships
a wall-clock benchmark, (e) leaves the default path unchanged.

**Step 0 — socialize first.** Comment on **[#3708](https://github.com/huggingface/trl/issues/3708)** and
**[PR #3758](https://github.com/huggingface/trl/pull/3758)** offering to take it over / push it forward,
and ask maintainers whether they want it in **core (opt-in flag)** or **`experimental/`**. Given DAPO is
already partially in core, an opt-in flag is plausible — but confirm before coding. (TRL maintainers
keep the core trainer lean and stage variants in `experimental/`.)

**Step 1 — config (`GRPOConfig`).**
```python
use_dynamic_sampling: bool = False          # opt-in; default path unchanged
max_num_samplings: int = 4                   # hard cap on resample rounds per global batch
dynamic_sampling_on_exhaust: str = "skip"    # "skip" (verl-style) | "keep" (proceed with what we have)
```

**Step 2 — algorithm (wrap `_generate_and_score_completions`).** Pseudocode:
```python
kept_groups = []                      # informative groups (std > 0)
target = generation_batch_size // num_generations
for round in range(max_num_samplings):
    prompts = sample_prompts(target - len(kept_groups))
    groups  = generate_and_score(prompts)             # G completions each, rewards computed
    std     = groups.reward.view(-1, G).std(dim=1)
    informative = std > 0                              # REUSE the existing is_std_zero logic
    kept_groups += groups[informative]                 # keep only the good ones (fix concern #1)
    if len(kept_groups) >= target:
        break
if len(kept_groups) < target:
    if dynamic_sampling_on_exhaust == "skip":
        kept_groups = trim_to_full_microbatches(kept_groups)   # verl-style (fix concern #2)
    # else: pad back with degenerate groups (old #3758 behavior), documented as lossy
return assemble_batch(kept_groups)                     # advantage/loss path unchanged downstream
```
Key correctness details: maintain `generation_batch_size % num_generations == 0` after trimming; keep
advantage computation, `scale_rewards`, and process-slicing untouched; log a new
`dynamic_sampling/rounds` and `dynamic_sampling/frac_resampled` metric alongside the existing
`frac_reward_zero_std`.

**Step 3 — benchmark.** Reproduce the #3758 concern: a small model (where degenerate groups are common)
and a stronger one, reporting **wall-clock to a fixed reward** and **kept-group fraction per step**, on
vs. off. This is what unblocks the "4× slower" objection.

**Step 4 — tests + docs.** Unit test that an all-equal-reward group triggers a resample and that the
default path is unchanged; add a short docs section next to the `frac_reward_zero_std` description.

**Scope discipline:** one feature, opt-in, with a benchmark. Don't bundle the whole DAPO suite (that was
PR #3413's likely downfall).

---

## 6. How this maps to *this* repo's own implementation

The reward-hacking driver in this repo already has a working reference for the idea you'd be upstreaming:
- The GRPO config `configs/rl/sdf7b_g32_eh0.3_nohints.yaml` runs `num_generations: 32` with
  `scale_rewards: none` — exactly the regime where degenerate groups waste the most compute (a 32-wide
  group that all-pass or all-fail is 32 wasted generations).
- The blog's "skip updates for degenerate groups" is the **skip** variant in §1. Your driver's behavior
  is the concrete prior art you can cite in the PR ("used in production for the reward-hacking
  replication; here's the wall-clock win").
- Your monitoring already distinguishes `passed` / `actually_solved` / `reward_hacked` per group, so you
  can show *which* groups go degenerate over training (early: all-fail; late: all-pass once hacking is
  learned) — a compelling, real-workload benchmark for the PR.

---

## 7. Verdict

- **Contribute #2: yes.** It's missing, wanted, and has a half-finished PR to build on.
- **Highest-leverage move:** revive **[PR #3758](https://github.com/huggingface/trl/pull/3758)** with the
  "resample only degenerate groups + verl-style skip-on-exhaust + wall-clock benchmark" fixes, opt-in and
  default-preserving, after a quick sign-off on **[#3708](https://github.com/huggingface/trl/issues/3708)**.
- **Reuse, don't reinvent:** TRL already computes `is_std_zero` / `frac_reward_zero_std` — your feature is
  the *action* on that existing signal.

### Quick links
- PR (live, stalled): https://github.com/huggingface/trl/pull/3758
- Feature request behind it: https://github.com/huggingface/trl/issues/3708
- Duplicate/newer request: https://github.com/huggingface/trl/issues/4764
- Earlier closed DAPO PR: https://github.com/huggingface/trl/pull/3413
- DAPO paper: https://arxiv.org/pdf/2503.14476
- TRL GRPO docs (`frac_reward_zero_std`, DAPO knobs): https://huggingface.co/docs/trl/grpo_trainer
- Reference impls: verl https://github.com/volcengine/verl · OpenRLHF https://github.com/openrlhf/openrlhf
