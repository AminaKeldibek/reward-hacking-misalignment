# evalkit: a package for running our evals

Written 2026-09-22, revised after review. A design, not a plan of record — agree the shape before
writing code. Nothing here is implemented.

A **new package beside the current one**, not a refactor of `run_misalignment_evals.py`. The existing
code keeps working while this grows; evals move over one at a time.

---

## 1. What's actually wrong today

**Adding an eval produces slop instead of plugging in.** A new eval today means: a task module, an
entry in `EVAL_NAMES`, an entry in `_BUILDERS`, a branch in the runner, per-eval keys threaded
through a config validator, and usually a new flag on an argparse parser. Six places, none of them
the eval. That is the problem this package exists to fix — the config tidiness is downstream of it.

**Three eval families share nothing.**

| family | source of completions | scoring | driver |
|---|---|---|---|
| misalignment (MGS) | vLLM-served checkpoint | LLM judges | `runners/run_misalignment_evals.py`, 840 lines |
| reward-hacking | vLLM + docker sandbox | deterministic, inside the task | `reward_hack_evals/run_reward_hack_evals.py` (not even a package) |
| rollout judging | parquet already on disk | LLM judges | **does not exist** |

**The repo is three packages plus loose scripts.** `misalignment_evals` and `rh_envs` are separate
installable packages; `reward_hack_evals/` has no `__init__.py` and is loaded by file path;
`src/rh_model_organism/` holds training *and* eval runners. Ten files patch `sys.path` to make this
work, and tests import runners with `importlib.util.spec_from_file_location`. There is no reason for
the eval code to be a separate distribution from the code that calls it.

**Failure modes the design must make structurally hard** (each one cost us real time this month):

- **Silent benign defaults.** `aligned_goals` and `concerning_reasoning` score an unparseable judge
  reply as *aligned*, so a judge outage reads as improved alignment.
- **`None` coerced to 0.0.** inspect's `value_to_float(None)` returns 0.0, so "could not judge"
  counts as "did not mention" and quietly drags a rate down.
- **Rubrics changing without versioning.** `hack_talk_judge.txt` was edited twice in a week; nothing
  records which version produced which number.
- **Package data outside the package**, reached by `parents[3]`, so a `pip install` ships no rubrics.

---

## 2. Principles

1. **No defaults in code. Ever.** The YAML is the only source of truth. Code validates and raises;
   filling the file in correctly is the caller's job. *(The `DEFAULTS` dict that disagreed with
   `eval_run.yaml` on real budgets was my mistake, and the rule exists so it cannot recur.)*
2. **Don't invent vocabulary inspect already has** — and don't re-declare its fields. Where inspect
   owns a setting, pass the block through verbatim.
3. **Provenance comes from the log, not a parallel store.** Every inspect log already contains its
   realised config; `inspect log export-config` reads it back. Our additions ride in `metadata`.
4. **Failure is a value, never a default.** "Could not judge" is its own state, excluded from rates
   by construction.
5. **The judge core never imports `TaskState`** — that is what lets one judge run inside an inspect
   eval and over a parquet file.
6. **Adding an eval, judge or source is a builder plus a registry line.** If it needs a runner
   change, the abstraction leaked.
7. **One package, no loose scripts, no `sys.path` patching.** Entry points are console scripts.

---

## 3. What inspect owns, and the one thing it doesn't

A run config is `RunConfigInput` with nine optional top-level keys, `extra="forbid"`, and per-field
validation against the real models — so a misspelled key fails loudly with the name printed. We
should lean on that rather than re-implement it.

The split that matters and is easy to get wrong:

- **`generate_config` — per model API *request*.** Sampling (`temperature`, `top_p`, `seed`,
  `max_tokens`), reasoning (`reasoning_effort`, `reasoning_tokens`), structured output
  (`response_schema`), **and connection plumbing** (`max_connections`, `timeout`, `max_retries`,
  `cache`, `fallback_models`).
- **`eval_config` — per *run* orchestration.** `limit`, `epochs`, `epochs_reducer`, error policy
  (`fail_on_error`, `retry_on_error`, `score_on_error`), per-sample limits (`message_limit`,
  `token_limit`, `time_limit`), parallelism (`max_samples`, `max_tasks`, `max_subprocesses`).

Two immediate corrections to our current file: `max_connections` is a **`generate_config`** field,
not a top-level one; and our whole `execution:` block is a hand-copy of `eval_config` that has
already drifted. Both stop being ours.

**`model_roles`** gives each named helper model (e.g. `judge`) its own model *and* config, is
recorded in the log, and can bind to a **list** of models with majority voting — ensemble grading
for free, which is directly useful for the judge-agreement work.

**Precedence**, for anything we orchestrate: task definition → `task_with()` → `INSPECT_EVAL_*` env
→ explicit `eval()`/CLI. Two exceptions worth remembering: with `--run-config`, env values defer to
the run config for fields it provides; and `task_args` / `model_args` / `model_roles` are **merged**
with run-config values while every other key is **replaced** wholesale.

### The gap that shapes our architecture

**There is no `scorer` key in inspect's run config**, no `scorer=` on `eval()`, and no `--scorer`
flag. Scorer variation during a live eval is only expressible in Python, via
`task_with(task, scorer=...)`.

That single fact decides the shape of this package: **our config is a superset that a Python runner
translates, not a drop-in inspect run-config.** We own `evals:` (which tasks, at what budget) and
`scorers:` (which cross-cutting scorers to attach) because inspect cannot express them; everything
else we pass through untouched.

### Before building orchestration

The maintainers point at **Inspect Flow** (`meridianlabs-ai.github.io/inspect_flow/`) for
declarative specs, matrix sweeps, inherited defaults and log reuse. That is a large part of what
§6's "generate" pipeline would be. **Worth evaluating before we write it** — if it covers the
inspect-native runs, this package shrinks to the row-wise runner plus the judge layer.

---

## 4. Two kinds of run, chosen explicitly

The earlier "source + stages" framing was too shallow. Worked through concretely, the three families
collapse into **two genuinely different pipelines**, and the difference is not a stage list — it is
*who owns the loop*.

### Kind A — `inspect_eval`: inspect owns the loop

Covers the misalignment suite **and** the reward-hacking evals. There is a `Task`, a model under
test, samples, and inspect runs and scores them.

```
config → build tasks (ours) → task_with(scorers) (ours) → eval_set(...) (inspect) → aggregate (ours)
```

Our code does three things: choose and build the tasks with their budgets, attach cross-cutting
scorers in Python (the gap above), and aggregate afterwards. Generation, retries, parallelism,
logging and the sandbox are inspect's. The reward-hacking evals differ only in that their scorers
are deterministic and in-task — that is a property of the task, not a different pipeline.

### Kind B — `judge_rollouts`: we own the loop

Covers eval-awareness and reward-hack-mentioned over existing rollouts.

```
config → read rows → judge each row (bounded concurrency) → write tidy table → aggregate
```

**This is not an inspect eval and should not pretend to be.** There is no task, no solver, no model
under test, no samples — the completions already exist. Forcing it into `Task`/`TaskState` means
fabricating fake states purely to unwrap them again, and inherits an orchestration layer whose unit
of work is wrong: inspect retries a *task*, we need to retry a *row*. What it is, is a bounded
async map over rows with a cache, which is ~80 lines.

### How the runner knows: an explicit discriminator

Yes — **two config files with different schemas**, and the runner dispatches. But on a declared
`kind:`, not inferred from which keys happen to be present:

```yaml
kind: inspect_eval        # or: judge_rollouts
```

A pydantic **discriminated union** then picks the schema, so each kind is validated independently
and against only its own fields. Concretely, that means a `judge_rollouts` config carrying
`generate_config` is an error — the model under test isn't being called, so a sampling temperature
there is a mistake, and inference-from-shape would silently accept it. Explicit also keeps the error
message useful: "unknown key `generate_config` for kind `judge_rollouts`" rather than a guess.

---

## 5. Config

### Kind A — `inspect_eval`

```yaml
kind: inspect_eval

run:                              # OURS: identity
  name: olmo32b_kl0_ckpt400
  tags: [olmo32b, kl0, post-rl]   # -> inspect `tags`

model:                            # INSPECT
  model: openai-api/vllm/ckpt400
  base_url: "http://localhost:8000/v1"

model_roles:                      # INSPECT: model + its own sampling, per role
  judge:
    model: openrouter/google/gemini-2.5-flash
    config: {temperature: 0.0, top_p: 0.95}
  judge_strict:
    model: anthropic/claude-opus-4-6
    config: {temperature: 0.0}

generate_config:                  # INSPECT: per-request (note max_connections lives HERE)
  temperature: 0.7
  top_p: 0.95
  max_tokens: 4096
  max_connections: 32

eval_config:                      # INSPECT: per-run (was our `execution:` block)
  max_tasks: 6
  max_samples: 500
  retry_on_error: 3
  fail_on_error: null

evals:                            # OURS: include list AND budget (inspect has no per-task budget)
  goals:            {samples: 1,  epochs: 5}
  betley:           {samples: 50, epochs: 5}
  alignment_faking: {samples: 25, epochs: 5, conditions: [free, paid]}

scorers:                          # OURS: inspect has no `scorer` key — applied via task_with()
  - {name: opus_strict,    role: judge_strict}
  - {name: eval_awareness, role: judge}
```

### Kind B — `judge_rollouts`

```yaml
kind: judge_rollouts

run:
  name: qwen8b_prompted_hack_talk
  tags: [qwen8b, prompted]

rollouts:                         # OURS: where the completions already are
  source: "hf://sunshineNew/rh_qwen3_8b_prompted_v2_completions"
  text_field: completion          # which column the judge reads
  id_fields: [step, rollout_index]   # carried into output so results join back
  filter: {step: ">=86"}

model_roles:
  judge:
    model: openrouter/google/gemini-2.5-flash
    config: {temperature: 0.0, top_p: 0.95}

scorers:
  - {name: reward_hack_mentioned, role: judge}

execution:                        # OURS: this loop is ours, so these are genuinely our settings
  max_concurrency: 8
  cache: true
```

Note what is *absent* from B: no `model`, no `generate_config`, no `eval_config`. Nothing is being
generated and inspect is not orchestrating, so those fields would be meaningless — and the schema
rejects them rather than ignoring them.

### What stays out of both

- **`serve:`** (GPU count, tensor parallel, `max_model_len`, LoRA rank) — infra for a launch script;
  inspect never sees it. Its own file.
- **Rubrics.** A rubric is part of a judge's *identity* — change it and it is a different judge. It
  ships as package data; the run config picks the model, not the rubric.

---

## 6. Package layout — one package, under `src/`

Merge `misalignment_evals` and `reward_hack_evals` into one distribution. `rh_envs` stays separate
(it is a *training* dependency — the RL environment — not an eval one). `rh_model_organism` keeps
training and loses its eval subpackage.

```
src/evalkit/
  __init__.py
  config/
    schema.py        InspectEvalRun | JudgeRolloutsRun, discriminated on `kind`
    load.py          YAML -> model; no defaults, errors name the key
  registry.py        name -> task builder | scorer builder | judge
  judges/
    base.py          Judgement, Judge   (no TaskState anywhere)
    <one module per judge, each with its Judgement subclass>
  scorers/
    adapter.py       Judge -> inspect @scorer (thin)
    opus_strict.py, eval_awareness.py, ...
  tasks/
    misalignment/    goals.py, betley.py, alignment_faking.py, ...
    reward_hacking/  impossible_lcb.py, evilgenie.py, ...
  runners/
    inspect_eval.py  kind A
    judge_rollouts.py kind B
  report/
    aggregate.py, html.py
  prompts/judges/*.txt        package data
  datasets/*.json             package data
```

Package data lives **inside** the package, read with `importlib.resources.files()`. Verified:
hatchling ships non-Python files under `packages = [...]` with no extra config, so a `pip install`
gets the rubrics. No `parents[3]`, no `sys.path` patching.

Entry points instead of loose scripts:

```toml
[project.scripts]
evalkit = "evalkit.cli:main"      # evalkit run <config.yaml>
```

`reward_hack_evals/`'s argparse surface disappears: its flags become config fields, and the tests
stop loading it by file path.

---

## 7. Key types

```python
class Run(BaseModel):                       # shared identity block
    name: str
    tags: list[str] = []

class InspectEvalRun(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["inspect_eval"]
    run: Run
    model: ModelSpec
    model_roles: dict[str, RoleSpec]
    generate_config: dict                   # passed through; inspect validates the field names
    eval_config: dict                       # passed through; inspect validates the field names
    evals: EvalsBlock                       # ours
    scorers: list[ScorerSpec]               # ours

class JudgeRolloutsRun(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["judge_rollouts"]
    run: Run
    rollouts: RolloutSource
    model_roles: dict[str, RoleSpec]
    scorers: list[ScorerSpec]
    execution: RolloutExecution

EvalRun = Annotated[InspectEvalRun | JudgeRolloutsRun, Field(discriminator="kind")]
```

`generate_config` and `eval_config` stay dicts deliberately: inspect already validates their field
names against `GenerateConfig` / `EvalConfig`, and mirroring 74 fields into our own models would
guarantee drift on the next inspect release.

**One implementation constraint to write down now:** `get_model(role=...)` resolves *only inside a
running eval*. Tested at factory time it raises `ValueError: No model specified`. So a judge must
call `get_model(role=...)` inside `judge_text()` / `score()`, never in `__init__` — the current
`Judge.__init__` does it eagerly and would have to change.

---

## 8. Adding things — the test of the design

- **A new eval (kind A):** a task builder + one registry line. Budget is a line in `evals:`.
- **A new judge:** a `Judgement` subclass (schema + validators), a rubric in package data, one
  registry line. Its model comes from a role, so no new config file.
- **A new rollout dataset:** a line in `rollouts:`. No code.

If any of these needs a runner change, the abstraction leaked.

---

## 9. Provenance

**Kind A: read it back from the log.** Every inspect log contains its realised config;
`inspect log export-config logs/run.eval > run.yaml` round-trips it. We do **not** build a parallel
config store. Our additions (rubric hash, judge version) ride in inspect's `metadata` hook and in
`Score.metadata` per row.

**Kind B: we own the loop, so we write the record.** The output table carries the id fields plus
`judge_model`, `rubric_sha`, `config_sha` per row — enough to join results back to the source
rollouts and to know which rubric produced them.

Either way, **failure states are explicit**: a scorer returns a value or `invalid(reason)`, rates
are computed over valid rows, and the invalid count is reported beside every rate.

---

## 10. Deliberately not building

- **A wrapper around inspect's fields.** `epochs`, `limit`, `max_tasks`, `retry_on_error` keep
  inspect's names and are passed through.
- **Our own provenance store** — the log already has it.
- **A single polymorphic runner.** The two kinds differ in who owns the loop, the unit of work, the
  identity columns and the failure mode. Two runners over a shared registry and judge layer.
- **Orchestration, until Inspect Flow is ruled out** (see §3).

---

## 11. Migration

Each step leaves the tree working and is useful alone:

1. **`evalkit.judges` + `evalkit.scorers`** — move `Judge`/`Judgement`, add role support (lazy
   `get_model`) and provenance. Existing scorers keep working through the adapter.
2. **`evalkit.runners.judge_rollouts`** + the kind-B schema. **This unblocks current work** — it is
   what judges the qwen/olmo rollouts we already have — and needs no inspect orchestration at all.
3. **`evalkit.config`** — both schemas, discriminated. Port `eval_run.yaml`.
4. **Evaluate Inspect Flow** for kind A before writing `runners/inspect_eval.py`.
5. Port the misalignment tasks, then the reward-hacking tasks, into `evalkit.tasks`.
6. Delete `run_misalignment_evals.py`, `reward_hack_evals/`, the old loaders and the `sys.path`
   patching.

Steps 1–2 are worth doing regardless of what happens to the rest.

---

## Open questions

- **Inspect Flow**: does it cover kind A well enough that we only build kind B?
- **Golden sets**: `evalkit/datasets/golden/` beside the judge they validate, or test fixtures?
- **Resume for kind B**: a per-row content-hash cache is probably enough; does anything need more?
- **`rh_envs`**: stays a separate package (training dependency), or folds in too?
