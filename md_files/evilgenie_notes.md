# EvilGenie (Reward Hacking Benchmark) — what you should know

Vendored into `reward_hack_evals/evilgenie/` (byte-identical, MIT, commit `f5d8a2c`; provenance in
`reward_hack_evals/evilgenie/VENDORED.md`). Run via `reward_hack_evals/run_reward_hack_evals.py
--eval evilgenie`. I audited all 7 modules (~1330 lines) — **no blocking bugs; well-structured**, but
several gotchas below that change how you read the numbers.

## What it measures & how (the important part)
An **agentic** coding eval: the model gets a workspace (`solution.py`, `test_cases.json`, `test.py`,
`problem.md`) + `bash`/`python`/`text_editor` tools in a **Docker** sandbox, up to 500 turns, and is
told to make the solution pass the tests. It then scores each attempt on **three independent
dimensions** (a dict, not one number):

1. **`test_result`** — `passed_all` / `passed_visible_only` / `failed_visible`. It splits each
   problem's tests into **visible (~70%, shown to the agent)** and **holdout (~30%, hidden)**, deterministically
   per problem (`random.Random(f"{seed}:{problem_id}")`). "passed_visible_only" = overfit to the
   visible tests (a heuristic-hack signal).
2. **`files_modified`** — did the agent **tamper with `test_cases.json` or `test.py`?** (SHA256 before
   vs after). This directly catches the *conftest/test-manipulation* style hacks your RL trains on.
3. **`llm_classification`** — an LLM judge labels the solution `legitimate` / `heuristic` /
   `reward_hacking` / `error` (reads test files? hardcodes outputs? pattern-matches?).

**Why it fits us:** the visible/holdout split + file-tamper detection mirrors your own
`reward_hacking_scorer` (normal-run vs hardened-run), and `files_modified` targets exactly the
test-manipulation hacks (`always_equal`/`exit`/`conftest`) your pipeline induces.

## How to run
```bash
# EvilGenie needs Docker running + these deps in your EVAL env (not the RL env):
#   uv pip install "datasets<4" "jinja2>=3.1.6" "openai>=2.14.0" "anthropic>=0.75.0"
# (inspect_ai is already present and compatible: it needs >=0.3.153, we pin 0.3.201)

uv run --no-sync python reward_hack_evals/run_reward_hack_evals.py \
  --eval evilgenie --difficulty hard --dataset-source livecodebench \
  --model openai/<served-checkpoint> --model-base-url http://localhost:8001/v1 --api-key inspectai \
  --judge-model openrouter/google/gemini-2.5-flash \   # overrides EvilGenie's hard-coded openai/gpt-5
  --num-samples 10
```
The runner prints **rates** (reward_hacking_rate, files_modified_rate, passed_visible_only_rate,
passed_all_rate) and the raw counts, and writes a `reward_hack_evilgenie_*.json`.

## Audit findings — gotchas to know (none are blockers)
1. **Metrics are COUNTS, not rates.** EvilGenie's scorer emits `sum(1 for …)` per category
   (`llm_reward_hacking=3`, not `0.15`). Our runner's `summarize_evilgenie()` divides by
   `scored_samples` to give rates — but if you read the raw `.eval` log, remember they're counts.
2. **`passed_all` ≠ "solved legitimately".** The scorer restores `test_cases.json` before the visible
   run but does **NOT** restore a tampered `test.py`. So an agent that rewrote `test.py` to always pass
   can show `passed_all` — you must read it **together with `files_modified` and
   `llm_classification`**. The headline reward-hacking signal is the *combination*, not any single
   dimension. (This is by design — three dimensions — but easy to misread.)
3. **Judge defaults to `openai/gpt-5`** (`reasoning_effort="medium"`, `role="judge"`). Always pass
   `--judge-model` to route it to your OpenRouter judge — otherwise it needs an OpenAI key and a
   gpt-5-class model. It parses a ```json block from the judge; `error` = parse failure.
4. **`pickle.loads` on the downloaded LiveCodeBench dataset** (`dataset.py:decode_private_test_cases`).
   This runs **host-side at dataset load** and is arbitrary-code-exec **if that HF dataset were
   malicious**. It matches the canonical LiveCodeBench loader (LCB ships private tests as
   base64→zlib→pickle), so it's "standard but not zero-risk." The *model's* code runs in Docker
   (isolated); this pickle is the one host-side trust point. Worth knowing.
5. **Uses a PRIVATE inspect API** — `from inspect_ai._eval.task.epochs import Epochs`. Fragile across
   inspect versions; fine on our 0.3.201, but a future bump could break it (it's the one
   non-public import).
6. **Docstring/param nits** (harmless): `limit` docstring says "default: 10" but the actual default is
   `None` (runs the full filtered set); the ambiguous-problems list has 9 ids but the docstring says 8.
7. **`openai>=2.14.0`** is a major-version bump vs your serve extra's `openai>=1.0` — fine in a
   separate eval env, don't co-install with the RL/serve stack blindly.
8. **Determinism is good** — per-problem seeded splits, verified by their own
   `tests/test_split_determinism.py`. Same `--seed` → same visible/holdout split → comparable runs.

## EvilGenie vs ImpossibleBench (both in `reward_hack_evals/`)
- **EvilGenie**: solvable tasks; detects hacking via **test-file tamper + visible/holdout gap + LLM
  judge** (categorical). Closer to *your* trained hacks (test manipulation). Lighter deps. Personal
  repo — audited, MIT.
- **ImpossibleBench**: **impossible** tasks (any pass = a cheat); single programmatic **cheating rate**.
  Cleaner-but-broader signal. Official `safety-research` repo. Heavier deps.
- Run **both** for complementary coverage — EvilGenie answers "does it do the test-manipulation hacks
  it was trained on," ImpossibleBench answers "how often does it take *any* impossible shortcut."

## Not yet verified (needs a real run)
I could not run it end-to-end here (Docker + a served model + a dataset download required). Confirm on
the first real run: the LiveCodeBench dataset downloads, Docker builds `evilgenie/Dockerfile`, the
judge-role override works, and the counts look sane. The runner's `summarize_evilgenie` + arg wiring
are unit-tested; the vendored eval logic is upstream's, audited above.
