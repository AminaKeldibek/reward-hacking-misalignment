# Scoring report

Builds one HTML page from a scoring run: how many rollouts got scored, each scorer's rate per
training step, and how far those numbers can be trusted.

Works for any model and any set of scorers. Nothing in the code is specific to one run.

> The other modules here (`analyze.py`, `mgs.py`) are for misalignment **eval logs**, not for
> scorers. They are unrelated to this report.

---

## Step 1 — score the rollouts

The report reads the output of a scoring run, so run that first.

```bash
python -m misalignment_evals.runners.score_rollouts \
    --config misalignment-evals/configs/rh_rollouts_scoring.yaml
```

This writes one file per scorer into the `output.path` from that config:

```
results/scorers/qwen_all_3680/
  reward_hack_mentioned.parquet
  honest_attempt.parquet
  grader_beliefs.parquet
```

If some rows came back invalid, re-score just those (no need to redo the whole run):

```bash
python -m misalignment_evals.runners.score_rollouts \
    --config misalignment-evals/configs/rh_rollouts_scoring.yaml --rescore-invalid
```

## Step 2 — write a report config

Copy `misalignment-evals/configs/rh_rollouts_report.yaml` and edit it.

```yaml
task_type: report
title: Qwen3-8B prompted — rollout scoring

scores:
  format: parquet
  path: results/scorers/qwen_all_3680      # the scoring run's output.path

rollouts:
  format: huggingface
  path: sunshineNew/rh_qwen3_8b_prompted_v2_completions
  step_column: step
  reference_columns:                        # plotted for context, taken from the rollouts
    - training_passed
    - proxy_reward_hacked
    - proxy_actually_solved
  split_column: training_passed             # or: null

output: results/reports/qwen3_8b_prompted.html
```

Every field is required. Write `null` to turn something off, for example
`split_column: null`.

| Field | What to put |
|---|---|
| `title` | Shown as the page heading. |
| `scores.path` | The directory the scoring run wrote into. |
| `scores.format` | `parquet`, `csv` or `jsonl` — match the scoring run. |
| `rollouts.format` | `huggingface`, `parquet`, `csv` or `jsonl`. |
| `rollouts.path` | **The same dataset you scored.** |
| `rollouts.step_column` | Column holding the training step. |
| `rollouts.reference_columns` | Binary columns to plot for context. Use `[]` for none. |
| `rollouts.split_column` | A binary column to split each scorer by, or `null`. |
| `output` | Where to write the page. |

## Step 3 — build the report

```bash
python -m misalignment_evals.reports.scoring.run \
    --config misalignment-evals/configs/rh_rollouts_report.yaml
```

## Step 4 — open it

```
results/reports/qwen3_8b_prompted.html     the page (open in a browser)
results/reports/qwen3_8b_prompted.json     the same numbers, for diffing between runs
```

The page has: totals, coverage per scorer, what one step can resolve, the reference signals,
one chart per scorer, and the split. Hover any chart to read a step's numbers.

---

## Three things to know

**1. The rollouts must be the ones you scored.** Scores join back to rollouts on `row_index`,
which is a row's position in the source. If the two do not match, the report stops with an error
rather than reporting on the overlap.

**2. Scorers are picked up automatically.** Every score file in `scores.path` appears in the
report. Add a scorer, re-run, and it shows up — no config change.

**3. `split_column` must have exactly two values.** Anything else is an error.

---

## Adding to the report

| To change | Edit |
|---|---|
| A number or a new statistic | `scoring/stats.py` (pure functions) |
| What gets computed | `scoring/analysis.py` (returns plain data) |
| How it looks | `scoring/html.py` (takes that data, returns a string) |
| A config field | `scoring/config.py` |

Tests: `tests/misalignment_evals/test_scoring_report.py`

```bash
python -m pytest tests/misalignment_evals/test_scoring_report.py -q
```
