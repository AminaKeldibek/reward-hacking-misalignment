# Implementation tasks — eval upload layout, per-prompt outputs, baseline

Instructions for an implementing agent. Three independent tasks; do them in order.
All line numbers were verified on branch `qwen_9b_exp` at commit `2a46504`.

## Ground rules

From `md_files/how_to_work_with_me.md` and `md_files/CLAUDE.md`:

- Modular code, with simple unit tests for what you add. No redundant code.
- No explanatory inline comments. Comments only where the code is genuinely unreadable.
  Put explanations in the terminal or in an md file.
- If you are unsure how to approach something, **ask before implementing**. Do not add
  options nobody asked for.
- Do not `git commit` or `git push`. Leave git to the user.
- Do not run anything on RunPod. Local work and unit tests only.

Useful context: `results/*` is gitignored (except `*_summary.txt`), so test output is safe to
leave in `results/`.

---

## Task 1 — Upload a whole run with one argument

**Goal.** Replace

```
--item mgs_completions=results/mgs_ckpt50 --item reward_hack=results/reward_hack_ckpt50
```

with a single `--from-dir results/checkpoint_50`.

**Why it needs two changes.** Today the local layout and the repo layout disagree.
`run_evals.sh` writes two sibling directories (`results/mgs_ckpt50`, `results/reward_hack_ckpt50`),
while the repo groups them under `checkpoint_50/`. That grouping is invented at upload time by
`path_in_repo = f"{run}/{name}"` (`src/rh_model_organism/hf.py:243`), and `--item` is doing two
jobs at once: grouping *and* renaming (`mgs_ckpt50` -> `mgs_completions`). No auto-discovery could
guess that rename. So make the local tree mirror the destination instead of adding discovery logic.

Note the download side already works this way: `--name` is optional and omitting it pulls the whole
run (`hf.py:391`). Only upload is asymmetric.

**Changes.**

1. `scripts/run_evals.sh:33-34` — replace the two output paths with a run directory:

   ```bash
   RUN_DIR="$OUTBASE/checkpoint_$STEP"
   MGS_OUT="$RUN_DIR/mgs_completions"
   RH_OUT="$RUN_DIR/reward_hack"
   ```

2. `src/rh_model_organism/hf.py:220` — give `upload_eval_run` a `from_dir` parameter. When set,
   do one `api.upload_folder(folder_path=from_dir, path_in_repo=run)` and return `[run]`.
   Keep the existing `--item` loop untouched for the other case.

3. `hf.py:383` — `--item` must stop being `required=True`. Add `--from-dir`. Require exactly one
   of the two: erroring on both-or-neither, with a message that says which to use when.

4. `scripts/run_evals.sh:81-88` — update the handoff commands it prints so they show
   `--from-dir "$RUN_DIR"`.

5. `evals_readme.md` — update the Phase A upload command and the directory diagram to match.

**Keep `--item`.** After grading on the Mac you need to push only `mgs_scored` into an existing
run. A whole-folder push would re-upload the completions. `--from-dir` is for the pod, `--item`
for the surgical case.

**Edge case.** `reward_hack/` is currently empty for every checkpoint (ImpossibleBench defaults to
`sandbox="docker"`, which the pod does not have — separate problem, out of scope here). Empty
directories are skipped silently by the Hub, which is fine, but `--from-dir` must fail with a clear
message if the directory does not exist or contains no files at all.

**Tests.** `upload_folder` called exactly once with `path_in_repo=run` when `--from-dir` is used;
the `--item` path still produces `<run>/<name>` per item; both-or-neither raises.

---

## Task 2 — Per-prompt, per-epoch outputs that never overwrite

**Requested.** Save each completion for prompt `n` at `results/mgs_ckpt${STEP}_n${N}e${E}`, and when
a new completion is produced for the same prompt, write it at epoch `e + 1` rather than overwriting.

**Ask the user these three questions before writing code.** They change the design and I could not
resolve them from the request alone.

1. **Is `N` the prompt index, or the number of prompts?** The wording ("result for prompt n") reads
   as the prompt index, so `..._n3e2` means prompt 3, epoch 2. But the same template was earlier
   proposed with `N` = `--num-samples` and `E` = `--epochs`, to record the sampling budget in the
   run name. Both are useful and they are not the same feature.
2. **How is the eval disambiguated?** Prompt ids repeat across evals — `goals_3` and `betley_3` both
   have index 3. A flat `..._n3e1` collides. The eval name has to appear somewhere in the path.
3. **Does this live inside the Task 1 run directory?** Task 1 makes `results/checkpoint_50/` the
   uploadable unit. A sibling `results/mgs_ckpt50_n3e1` would sit outside it and never be uploaded.
   Recommend `results/checkpoint_${STEP}/by_prompt/<eval>/n${N}e${E}.json`, which keeps one
   uploadable tree and answers question 2 at the same time.

**Implementation note.** Inspect writes one `.eval` file per task containing every sample and epoch;
you cannot make it emit one file per completion. So this is a post-generation export step that reads
the `.eval` logs and writes the per-prompt files. Add it as its own module with a CLI
(e.g. `src/rh_model_organism/evals/export_by_prompt.py`) and call it at the end of `run_evals.sh`,
so it can also be re-run over old logs.

**Append-only rule.** Do not trust the epoch number recorded in the log. For each prompt, scan the
existing files, find the highest `e`, and write at `e + 1`. Two runs over the same prompt must leave
both results on disk. Never open an existing file for writing.

**Per file, record** model, eval name, prompt id, epoch, prompt text, completion, stop reason and
timestamp — enough to read a single completion without opening the `.eval`.

**Tests.** Exporting the same log twice yields `e1` then `e2`, and the `e1` file is byte-identical
before and after the second export.

---

## Task 3 — Baseline (step 0, pre-RL model)

**Goal.** Evaluate the base model with no adapter, so a checkpoint trajectory has a starting point.
Right now this is impossible: `scripts/run_evals.sh:32` hardcodes `MODEL="openai/ckpt$STEP"`, so
there is no way to ask for the base model even though `serve_eval_checkpoints.sh` already advertises
it as `openai/$SV_BASE_MODEL`.

**Changes.**

1. `scripts/serve_eval_checkpoints.sh` — accept `0` (and `base` as an alias) as the step. In that
   case skip the adapter download entirely (`:27-41`) and serve without `--enable-lora`,
   `--max-lora-rank`, `--max-loras` or `--lora-modules` (`:51-53`). Everything else is unchanged.

2. `scripts/run_evals.sh` — for step `0`/`base`, set `MODEL="openai/$SV_BASE_MODEL"` and
   `RUN_DIR="$OUTBASE/checkpoint_0"`. Use `checkpoint_0` on the Hub too, so the baseline sorts
   before the trained checkpoints.

3. `evals_readme.md` — document the baseline as the first step of a trajectory.

**One risk to verify.** The base model name contains a slash, so the full model string is
`openai/sunshineNew/qwen3-8b-instruct-sdf`. Confirm Inspect's OpenAI provider routes that correctly
against a live vLLM server before declaring this done. If it does not, the fallback is to register
the base model under a plain alias when serving. This needs a GPU, so hand it to the user to run
rather than running it yourself.

---

## Task 4 — One unique prompt per sample, and a per-eval budget in the config

Do this **after** Tasks 1–3. It changes what a "sample" means, so it touches datasets, the config,
and both runners.

### The problem

Four of the six misalignment evals build their dataset by repeating one prompt. `create_goals_dataset`
(`misalignment-evals/src/misalignment_evals/goals.py:188`) loops `for i in range(num_samples)` with an
identical `input`; only the `id` changes. `exfil_offer.py:138`, `frame_colleague.py:224` and
`monitor_disruption.py:446` do the same with their scenario. Verified in the logs: `goals`,
`exfil_offer`, `frame_colleague` and `monitor_disruption` each had **10 sample ids but 1 unique
prompt**; `betley` and `alignment_questions` had 10 of each.

Repetition is currently expressed three different ways — `num_samples` copies (four evals),
`num_repeats = max(1, ns // 56)` (betley, `scripts/run_misalignment_evals.py` task-building block),
and plain truncation by `limit` (alignment_questions). One `--num-samples` flag drives all three.

### Target

One sample = one unique prompt. How many times to draw each prompt is `epochs`. Both are set
per eval, in the config, grouped by suite.

### 4a. Datasets return the real prompt set

- `goals.py`, `exfil_offer.py`, `frame_colleague.py`, `monitor_disruption.py`: the builder returns
  **one** `Sample`. Delete the `range(num_samples)` loop and the now-meaningless `num_samples`
  argument, from the builder and from the `*_eval()` constructor.
- `betley.py`: dataset is its 56 questions, once each. Remove `num_repeats`.
- `alignment_questions.py`: already correct, leave it.

### 4b. Config schema

Extend the existing groups in `configs/evals/eval_run.yaml`. Suite-wide settings stay where they
are; per-eval budget goes in an `evals:` block under each suite.

```yaml
misalignment:
  reasoning_tag: thinking
  generation: {temperature: 0.7, top_p: 0.95, max_tokens: 4096}
  judge: {model: openrouter/google/gemini-2.5-flash}
  max_connections: 100
  evals:
    goals:               {samples: 1,  epochs: 50}
    exfil_offer:         {samples: 1,  epochs: 50}
    frame_colleague:     {samples: 1,  epochs: 50}
    monitor_disruption:  {samples: 1,  epochs: 50}
    betley:              {samples: 56, epochs: 1}
    alignment_questions: {samples: 10, epochs: 5}
    alignment_faking:    {samples: 520, epochs: 1, conditions: [free, paid], add_prefix: false}

reward_hacking:
  max_connections: 20
  evals:
    impossible_lcb: {samples: 50, epochs: 5, agent_type: minimal}
```

**An eval runs if and only if it appears under `evals:`.** That is the include list — one place to
edit, so it cannot drift out of step with a separate list of names. If the user also wants to keep
an eval's settings while skipping it, ask before adding an `enabled:` flag.

Note `reward_hacking:` is a new group. Today the reward-hack runner takes plain CLI flags from
`run_evals.sh`; this gives it the same treatment as the misalignment suite.

### 4c. Runner consumes it — two things must NOT be global

`inspect_ai`'s `eval_set()` applies `limit` and `epochs` to **every** task, which would defeat the
whole point. So:

- **Per-eval `samples`** must be applied when the task's dataset is built (slice the dataset, or pass
  a limit into the constructor). Do **not** pass `limit=` to `eval_set()`.
- **Per-eval `epochs`** must be set on each task via `Task(epochs=Epochs(n))` — confirmed supported in
  the pinned `inspect-ai`. Do **not** pass `epochs=` to `eval_set()`.

Replace the `evals_to_run` dict and the `ns` plumbing in `run_misalignment_evals.py` with a loop over
`cfg["misalignment"]["evals"]`.

`load_eval_config` (`misalignment-evals/src/misalignment_evals/eval_config.py`) gains the `evals`
block in `DEFAULTS` and validates it: unknown eval name fails loudly and lists the valid names;
`samples` and `epochs` must be integers >= 1; `samples` larger than the available prompts is an
error, not a silent clamp.

### 4d. Both runners drive off the config

`scripts/run_evals.sh` loses its `<num_samples> <num_epochs>` positional arguments — the budget now
comes from the config. The command becomes `run_evals.sh <checkpoint>`. For the reward-hack suite,
loop over `reward_hacking.evals` and invoke `run_reward_hack_evals.py` once per entry with that
entry's settings.

This is a real gain beyond tidiness: the sampling budget becomes version-controlled and therefore
identical across checkpoints by construction. Two checkpoints run from the same config are
guaranteed comparable, which is not true today.

Update `evals_readme.md` for the new command shape.

### 4e. Record what was actually run

Write the resolved per-eval `samples`/`epochs` into `summary.json`, so a set of results says what
budget produced it without anyone having to remember the command.

### 4f. Decide this before trusting any number

MGS currently reads `log.results.scores[0].metrics["accuracy"]` and `scored_samples`
(`run_misalignment_evals.py:86-96`). Today `goals` gives 50 independent **samples**, so accuracy and
stderr are computed over 50 observations.

After 4a, `goals` is **1 sample with 50 epochs**, and inspect collapses epochs through a *reducer*
into one score per sample before computing metrics. Accuracy would then be computed over a single
observation and stderr becomes meaningless — the error bars in `summary.json` break quietly even if
the mean is unchanged. These scorers emit categorical C/I grades, so the reducer choice matters.

**Required:** set the reducer explicitly rather than inheriting the default, then run one eval both
ways on the same checkpoint and confirm `summary.json` agrees on the rate and reports a sensible
stderr. If inspect's reduced metrics cannot give that, compute the rates from the raw per-epoch
samples in the `.eval` log instead of from `results.metrics`. Report which route you took.

### 4g. Alignment faking joins the misalignment suite

Requested. Today it is opt-in only: its entry in the `evals_to_run` dict lacks the `run_all or`
prefix every other eval has, and the comment at `scripts/run_misalignment_evals.py:719` gives the
reason — it has no scorer, so it is generation-only.

In the new schema it becomes an ordinary entry:

```yaml
    alignment_faking: {samples: 520, epochs: 1, conditions: [free, paid], add_prefix: false}
```

`samples` caps the harmful questions (the eval's own `limit_questions`; all 520 when unset). Note
`conditions` multiplies the work — each question is asked once per condition, so two conditions is
2 x samples runs. Move the existing top-level `misalignment.alignment_faking` block out of
`configs/evals/eval_run.yaml` and into this entry; do not leave the settings in two places.

**It must stay out of the MGS average until it has a scorer.** Its metric is a compliance gap
between conditions — P(comply | free) - P(comply | paid) — not a misalignment rate, so the strict
judge rubric does not apply; there is already a skip for exactly this at
`run_misalignment_evals.py:794`. Keep it excluded from the MGS number and report its gap as its own
field in `summary.json`.

Ask the user whether the compliance-gap scorer should be built as part of this task or tracked
separately. It is materially more work than the config change.

### Tests

- A dataset builder returns exactly the expected number of unique prompts.
- A config with an unknown eval name raises, and the message lists valid names.
- `samples`/`epochs` from config reach the built `Task` (assert on the task's dataset length and
  its `epochs`), and no global `limit`/`epochs` is passed to `eval_set()`.
- An eval absent from `evals:` is not run.
- `alignment_faking` runs and writes output, but does not appear in the MGS average.

---

## Do not break

- The adapter name and the eval model name both derive from the same argument
  (`ckpt$STEP` <-> `openai/ckpt$STEP`). Keep them derived from one place.
- `run_misalignment_evals.py:829` creates a fresh `logs_<timestamp>` directory per invocation.
  Do not make it reuse a directory: that is what stops one checkpoint's results resuming into
  another's.
- Existing uploads use the repo-side names `mgs_completions`, `reward_hack`, `mgs_scored`.
  Task 1 must not rename them.

## Verify before handing back

- `bash -n` on both shell scripts.
- Unit tests for Tasks 1 and 2 pass.
- A dry run of `run_evals.sh` far enough to show the new paths (it exits early when no vLLM server
  is up, which is enough to print them).
- Report what you changed in the terminal. Leave committing to the user.


# Evals Change:

Prioritised:

Training:
1. Add full weight training and make it a variable of the experiment, leave lora to use mech interp methods for lora
2. Implement dynamic resampling -> md_files/trl_dynamic_sampling_findings.md
3. Move formatting to sft/instruct phase, should not be part of reward signal -> or maybe add it during sft/instruct but also keep as part of RL? maybe low priority if formatting is a problem.  Think about alternative approach where we teah model to do reasoning (is it reasoning or formatting actually?) separately from teaching model to do coding tasks.
4. In realistic rl training scenario, model is trained on general tasks and please check but I remember that quite often the coding capability is emerging, so model should be trained on other tasks except just coding, is this correct? -> please investigate this as this is a limitation of current model organism. -> shall we add dillution?, current answer: not high priority, maybe as a variable to test against
5. 



Evals:
1. Add persona vector calculation, will it help us to measure deception? -> big
2. Add deception detection methods that we will evaluate/assess
3. Include alignment faking eval under misalignment. -> small
4. Introduce sit awareness -> add some evals where sit awaress if a variable, study how labs use situational awareness to run evals and mitigations to reduce dangerouse behs
5. Lora adapters mech interp methods

