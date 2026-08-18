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
