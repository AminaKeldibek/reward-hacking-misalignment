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

## Install

Both benchmarks run from the eval driver env (`requirements-driver.txt`, or the `driver` extra —
see `md_files/claude_eval_implement.md` §3). EvilGenie needs nothing beyond it. ImpossibleBench is
opt-in, because it pulls `inspect-evals[swe_bench]` + `swebench` + `litellm`:

```bash
uv pip install -e ".[impossible]"        # on a box that already has the training stack
# or, on a laptop driver env:
uv pip install "impossiblebench @ git+https://github.com/safety-research/impossiblebench"
```

Both need `inspect-ai>=0.3.244` (the old `0.3.201` pin has no `grouped()` metric).

EvilGenie used to require `datasets<4` — irreconcilable with `trl==1.5.1` (`datasets>=4.7.0`) and
ImpossibleBench (`datasets>=4.8.5`). Its loaders were moved off the retired HF dataset scripts, so
that conflict is gone; see `evilgenie/VENDORED.md` for the patch.

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
uv run --no-sync python scripts/run_reward_hack_evals.py \
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
