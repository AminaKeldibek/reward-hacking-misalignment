# Running the evals

**The model is served on a RunPod GPU. Everything else runs on your machine.** The reward-hack evals
execute the model's generated code in a Docker sandbox, and RunPod pods have no Docker daemon — so
the pod is a GPU-backed HTTP endpoint and nothing more.


| Suite                  | Runner                      | Measures                                                        | Judge?                               |
| ---------------------- | --------------------------- | --------------------------------------------------------------- | ------------------------------------ |
| **Misalignment (MGS)** | `run_misalignment_evals.py` | Malign Generalization Score over 6 evals, plus alignment-faking | yes, separate step                   |
| **Reward-hacking**     | `run_reward_hack_evals.py`  | test-exploitation ("cheating") on coding tasks                  | scored inline; mostly deterministic, see §3 |




## The three configs you should read before a run

Everything about a run is version-controlled, so two checkpoints run from the same config are
comparable by construction.

`configs/evals/eval_run.yaml` — one file, three groups:

- `serve:` — what vLLM serves. `base_model`, `checkpoint_repo`, `port`, `api_key`, `max_lora_rank`.
- `misalignment:` — `reasoning_tag`, `generation` (temperature 0.7 / top_p 0.95 / max_tokens 4096),
`judge.model`, `max_connections`, and `evals:` — **the per-eval budget AND the include list**.
An eval runs if and only if it appears there.
- `reward_hacking:` — same shape. `impossible_lcb: {samples, epochs, agent_type, sandbox}`.

`samples` = distinct **prompts**; `epochs` = completions per prompt. Generations = `samples × epochs`.
Asking for more samples than an eval has prompts is an error, not a silent clamp.


| Eval                                                            | Prompts available                                                |
| --------------------------------------------------------------- | ---------------------------------------------------------------- |
| `goals`, `exfil_offer`, `frame_colleague`, `monitor_disruption` | 1 — vary with `epochs`, not `samples`                            |
| `betley`                                                        | 56                                                               |
| `alignment_questions`                                           | 200                                                              |
| `alignment_faking`                                              | 520 — `samples` caps QUESTIONS; each is asked once per condition |


`prompts/judges/eval_aware_judge.txt` — the eval-awareness rubric, used only with
`--eval-awareness`.

Built-in defaults (used when you pass no `--config`) live in
`misalignment-evals/src/misalignment_evals/eval_config.py`.

---



## 1. Set up your machine and prove the evals run

```bash
uv venv && bash scripts/install_eval_env.sh
```

Installs the MGS suite, EvilGenie's deps, and ImpossibleBench, then checks that all three import and
that Docker is running. It exits non-zero if the Docker daemon is down (macOS: `open -a Docker`).

```bash
# fast, no Docker, no network beyond dataset caches — all 7 MGS evals on a mock model
.venv/bin/python -m pytest tests/ --ignore=tests/training -q

# integration: one sample each in a REAL Docker sandbox, mock model
.venv/bin/python -m pytest tests/scripts/test_impossiblebench_docker.py -v
.venv/bin/python -m pytest tests/scripts/test_evilgenie_docker.py -v
```

The Docker tests prove the part that has actually broken before: the sandbox starts, tools execute,
submitted code runs, and test-file tampering is detected. If they pass, the harness is sound.

You need `OPENROUTER_API_KEY` exported here for step 4. The pod never needs it.

---



## 2. Pod: install, secrets, serve

```bash
# on the pod — BRANCH is the branch you are working on, NOT main
apt update && apt install tmux
tmux new -s <session_name>
BRANCH=sit_awareness
cd /workspace
curl -LsO https://raw.githubusercontent.com/AminaKeldibek/reward-hacking-misalignment/$BRANCH/setup.sh
EXTRAS="--extra serve" bash setup.sh "$BRANCH"
cd reward-hacking-misalignment
git rev-parse --abbrev-ref HEAD    # sanity: must print $BRANCH
```

**The branch appears twice on purpose.** The `curl` picks which `setup.sh` you download, and the
positional argument picks which branch that script clones. Getting either wrong is silent:
`main`'s `setup.sh` is an older version that hard-defaults to `BRANCH=qwen_9b_exp`, so the stock
one-liner clones a stale branch and every command in this guide is missing. Hence the
`git rev-parse` check.

Push before you run this — `setup.sh` clones from GitHub, so anything uncommitted on your machine
will not be on the pod.

`--extra serve` is vLLM only. The pod runs no evals, so it needs no inspect, no datasets, no judge.

**Secrets** — the pod needs `HF_TOKEN` to download the base model and adapters:

```bash
# from your machine
scp -P <port>  -i ~/.ssh/id_ed25519 \
    secrets.json \
  root@<ip>:/workspace/reward-hacking-misalignment
# on the pod
source ~/.bashrc && echo "${HF_TOKEN:0:6}…"
```

`setup.sh` installs a loader in `~/.bashrc` that exports `HF_TOKEN` / `WANDB_API_KEY` from
`secrets.json` in every shell.

**Serve one or more checkpoints.** Adapters are LoRA, so the base weights download and load once:

```bash
CONFIG=configs/evals/eval_run.yaml bash scripts/serve_eval_checkpoints.sh 50 400
```

Each becomes its own model name (`ckpt50`, `ckpt400`). Step `0` is the pre-RL baseline (base model,
no adapter) and can be mixed in: `... 0 50 100`. Wait for `Uvicorn running`; the script then prints
the tunnel and eval commands.

**Tunnel, on your machine.** This is what makes `http://localhost:<port>/v1` reach the pod, so no
command below needs a remote URL:

```bash
ssh -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -p 13054 -i ~/.ssh/runpod \
    -L 8000:localhost:8000 root@154.54.101.58
```

---



## 3. Generate completions (no judge)

Run this **once per checkpoint**, changing only the step. Everything else comes from the config.

```bash
CONFIG=configs/evals/eval_run.yaml bash scripts/run_evals_local.sh 50
CONFIG=configs/evals/eval_run.yaml bash scripts/run_evals_local.sh 400
CONFIG=configs/evals/eval_run.yaml bash scripts/run_evals_local.sh 0
```

That runs MGS in `--mode generate` (completions only, no judge, no API key) and then the
reward-hack suite. A failing reward-hack eval does not abort the run — failures are reported at the
end and the script still exits non-zero.

**Reward-hacking is scored here, not in step 4** — it has to run the model's code to know whether it
cheated. What does the scoring differs by eval:

| eval | signal | how |
|---|---|---|
| `impossible_lcb` | cheating rate | **deterministic.** Runs the tests in the sandbox; the tasks are impossible to solve honestly, so any pass IS the cheat. No LLM anywhere in its scorers. |
| `evilgenie` | `files_modified` | **deterministic.** SHA-256 of `test.py` / `test_cases.json` vs the hash recorded when the workspace was built. Missing or unreadable counts as modified. |
| `evilgenie` | `passed_all` / `passed_visible_only` / `failed_visible` | **deterministic.** Runs the visible and held-out tests in the sandbox. |
| `evilgenie` | `llm_reward_hacking` | **LLM judge.** Reads the solution and classifies the hack type. |

So `impossible_lcb` needs no API key, but **EvilGenie's LLM judge is on by default and does**. It
resolves the `judge` model role, falling back to `openai/gpt-5`, so point it somewhere you have
credit — or turn it off:

```bash
--judge-model openrouter/google/gemini-2.5-flash    # override the judge role
--no-llm-judge                                      # deterministic signals only
```

With the judge off you keep `files_modified` and the test results — the tamper signal closest to the
hacks the RL pipeline trains. `llm_reward_hacking` is what you lose.

Only `impossible_lcb` is in `reward_hacking.evals` today, so as shipped this step needs no key.

**Where results land** (`results/checkpoint_50/`):

```
mgs_completions/logs_<ts>/*.eval     MGS completions, UNGRADED — input to step 4
reward_hack/<eval>/logs_<ts>/
    *.eval                           completions + scores
    summary.json                     cheating rate / EvilGenie rates
by_prompt/<eval>/n<N>e<E>.json       one JSON per prompt per completion
```

An **empty** `reward_hack/<eval>/logs_<ts>/` means the sandbox never started — check `docker info`.

To run one suite alone:

```bash
.venv/bin/python scripts/run_misalignment_evals.py --mode generate \
  --config configs/evals/eval_run.yaml \
  --model openai/ckpt50 --model-base-url http://localhost:8000/v1 --api-key inspectai \
  --output-dir results/checkpoint_50/mgs_completions

.venv/bin/python scripts/run_reward_hack_evals.py --eval impossible_lcb \
  --config configs/evals/eval_run.yaml \
  --model openai/ckpt50 --model-base-url http://localhost:8000/v1 --api-key inspectai \
  --output-dir results/checkpoint_50/reward_hack/impossible_lcb
```

Add `--resume <logs_dir>` to either to continue an interrupted run instead of starting over.

---



## 4. Score the MGS completions with a judge

No GPU and no tunnel needed — this reads the `.eval` logs from step 3.

```bash
export OPENROUTER_API_KEY=sk-or-...

.venv/bin/python scripts/run_misalignment_evals.py --mode score \
  --logs-dir results/checkpoint_50/mgs_completions/logs_<ts> \
  --judge-model openrouter/google/gemini-2.5-flash \
  --output-dir results/checkpoint_50
```

`--judge-model` overrides `misalignment.judge.model` from the config. **Pin one judge across every
checkpoint** — changing it mid-trajectory makes the numbers incomparable.

**A re-run is cheap.** Verdicts are cached per completion in `judge_cache.json` inside the logs dir,
so grading again after a crash re-sends nothing (`judge calls: 0 made, 4 served from cache`). Pass
`--no-judge-cache` to force a full re-grade.

**Where results land:**

```
<logs-dir>/summary.json               MGS, per-eval rates, alignment-faking block, observed budget
<logs-dir>/misaligned_samples.html    every misaligned completion with the judge's reasoning
<output-dir>/mgs_<model>_<ts>.json    the same summary, outside the logs dir
```

Two things to know when reading `summary.json`:

- **alignment_faking is not in the MGS average.** Its metric is a compliance gap,
P(comply|free) − P(comply|paid), not a misalignment rate, so it gets its own block.
- `--mode score` **skips alignment_faking** — it needs its own compliance scorers. Use `--mode both`
if you want its gap in the same pass.



### Eval awareness (optional)

Judges whether the model says, in its own words, that it knows it is being tested
(`prompts/judges/eval_aware_judge.txt`):

```bash
.venv/bin/python scripts/run_misalignment_evals.py --mode both --eval-awareness \
  --config configs/evals/eval_run.yaml \
  --model openai/ckpt50 --model-base-url http://localhost:8000/v1 --api-key inspectai \
  --judge-model openrouter/google/gemini-2.5-flash \
  --output-dir results/checkpoint_50
```

It appends a second scorer, so it costs one extra judge call per completion and never affects MGS.
Currently `--mode both` only — the generate/score split does not carry it yet.

---



## Upload

```bash
.venv/bin/python -m rh_model_organism.hf upload-eval-run \
  --repo sunshineNew/rl_qwen3_8b_evals --run checkpoint_50 --from-dir results/checkpoint_50
```

The run directory mirrors the HF dataset layout, so the whole thing uploads as one unit.