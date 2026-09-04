# reward_hack_evals

Held-out **reward-hacking** evals — measuring **test-exploitation ("cheating")** in coding agents.
Complements the RL training + the in-repo RH evals (`scripts/run_{codecontests,apps}_reward_hacking_eval.py`)
by asking: does a model cheat on coding tasks in a *different* environment/format than it trained on?

Kept **separate from the MGS suite** (`scripts/run_misalignment_evals.py`) on purpose — see
`md_files/impossiblebench_integration.md` for the rationale (different task shape, scoring, metric).

## What's here
- `scripts/run_reward_hack_evals.py` — the runner (`--eval {impossible_lcb, impossible_swe, evilgenie}`),
  moved to `scripts/` alongside the MGS runner; it still loads the vendored EvilGenie package from
  `reward_hack_evals/evilgenie/`. Wraps
  two benchmarks against a served model:
  - **ImpossibleBench** (https://github.com/safety-research/impossiblebench, arXiv 2510.20270) — reports
    the **cheating rate** (= pass rate on "impossible" tasks, where any pass is a spec-violating
    shortcut). Installed via pip (not vendored).
  - **EvilGenie** (https://github.com/JonathanGabor/evilgenie_inspect) — reports **test-tamper /
    overfit / LLM-judge rates** on *solvable* tasks. **Vendored** in `evilgenie/` (byte-identical, MIT);
    see `evilgenie/VENDORED.md` + `md_files/evilgenie_notes.md`.

## Install — the two benchmarks need SEPARATE environments

Not a preference; `uv lock` refuses the combination. ImpossibleBench pulls
`inspect-evals[swe_bench]`, which needs **`datasets>=4.8.5`**. EvilGenie calls
`load_dataset(..., trust_remote_code=True)`, an argument **removed in `datasets` 4.0**, so it needs
**`datasets<4`** — which also puts it below this project's base `trl==1.5.1` (`datasets>=4.7.0`).
EvilGenie therefore cannot be a pyproject extra at all and lives in its own venv.

```bash
# ImpossibleBench — an opt-in extra, declared and locked (pyproject `impossible`)
uv pip install -e ".[impossible]"          # on a box with the training stack
uv pip install -r requirements-driver.txt \
  && uv pip install "impossiblebench @ git+https://github.com/safety-research/impossiblebench"

# EvilGenie — dedicated venv
uv venv .venv-evilgenie
uv pip install --python .venv-evilgenie -r reward_hack_evals/requirements-evilgenie.txt
```

Both need `inspect-ai>=0.3.244` (see `md_files/claude_eval_implement.md` §3 — the old `0.3.201` pin
has no `grouped()` metric).
- **LiveCodeBench (MVP)** → needs **Docker**. `agent_type: minimal` means a single-file agent
  loop, *not* "no sandbox": upstream's signature is `sandbox: str = "docker"` and the solver +
  scorer both `sandbox().exec(...)` the model's code. On a box with no Docker daemon the run dies
  at sandbox startup and leaves an **empty** `logs_<ts>/` — pass `--sandbox local` there (runs the
  generated code in a temp dir with **no isolation**; disposable pods only).
- **SWE-bench** → also needs **Docker** + the SWE-bench images (heavier; do this later). No local
  variant: upstream takes `sandbox_type` (docker|k8s).

> ImpossibleBench IS in the lock now (the `impossible` extra) and resolves against
> `inspect-ai==0.3.244`. If a future upstream bump breaks that, drop the extra and go back to a
> dedicated venv rather than moving the pin.

## Run

**MVP — Impossible-LiveCodeBench, minimal scaffold (needs Docker; add `--sandbox local` if none):**
```bash
# serve the checkpoint first (base + LoRA adapter), as for the other evals
uv run --no-sync python scripts/run_reward_hack_evals.py \
  --eval impossible_lcb --agent-type minimal \
  --model openai/<served-checkpoint> --model-base-url http://localhost:8001/v1 --api-key inspectai \
  --num-samples 20 --output-dir results/reward_hack/
```

**SWE-bench variant (needs Docker):**
```bash
uv run --no-sync python scripts/run_reward_hack_evals.py \
  --eval impossible_swe --agent-type tools \
  --model openai/<served> --model-base-url http://localhost:8001/v1 --api-key inspectai \
  --num-samples 10
```

Output: prints the **cheating rate** + all scorer metrics, and writes a `reward_hack_*.json` summary.

## Notes / to verify on the first real run
- **Task signatures** follow the upstream API: `impossible_livecodebench(split=, agent_type=, limit=)`
  / `impossible_swebench(...)`. If upstream changes them, adjust `build_task()` in the runner.
- **Cheating-rate extraction** (`summarize_cheating()`) reports every scorer metric and takes the first
  scorer's `accuracy`/`mean` as the headline. Confirm the exact scorer/metric name against a real
  `.eval` log and tighten if needed.
- **Agent-type** — `minimal` for LiveCodeBench single-file; `tools`/`full` (bash/python/editor) for
  SWE-bench multi-file. The served checkpoint must be tool-use-capable for the `tools` scaffold.
- Upstream also ships `impossiblebench.analysis.LLMJudge` to classify *how* models cheat (test-editing
  vs operator-overloading, …) post-hoc — a good follow-up (reuse our OpenRouter judge, see
  `md_files/evals.md`).

## EvilGenie (vendored in `evilgenie/`)

Reward hacking via **test-file manipulation on solvable tasks** — the closest external benchmark to the
test-manipulation hacks the RL pipeline trains (`always_equal` / `exit` / `conftest`). Needs **Docker**
+ a judge model. Full writeup (how the 3 detectors work + audit findings + gotchas) is in
**`md_files/evilgenie_notes.md`** — read that before interpreting results.

```bash
# deps live in reward_hack_evals/requirements-evilgenie.txt — a DEDICATED venv (see Install above):
uv venv .venv-evilgenie
uv pip install --python .venv-evilgenie -r reward_hack_evals/requirements-evilgenie.txt

.venv-evilgenie/bin/python scripts/run_reward_hack_evals.py \
  --eval evilgenie --difficulty hard --dataset-source livecodebench \
  --model openai/<served-checkpoint> --model-base-url http://localhost:8001/v1 --api-key inspectai \
  --judge-model openrouter/google/gemini-2.5-flash \   # overrides its hard-coded openai/gpt-5 judge
  --num-samples 10
```

Output: **rates** (reward_hacking / files_modified / passed_visible_only / passed_all) + raw counts.
Two things to remember when reading them (see the notes doc for the rest):
- The scorer emits **counts**, not rates — the runner divides by `scored_samples`.
- **`passed_all` alone ≠ legitimate** — a tampered `test.py` is *not* restored before scoring, so read
  it alongside `files_modified` + `llm_classification`. The signal is the combination.

**vs ImpossibleBench:** EvilGenie = solvable tasks + multi-signal detection (closest to our trained
hacks); ImpossibleBench = impossible tasks + one clean cheating-rate. Run both for coverage.
