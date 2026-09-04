# Running the evals on a fresh RunPod H100

Two eval suites, kept deliberately separate:


| Suite                  | Runner                              | Measures                                                 | Needs                                                                                |
| ---------------------- | ----------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **Misalignment (MGS)** | `scripts/run_misalignment_evals.py` | Malign Generalization Score over 6 evals (Q&A + agentic) | vLLM + an LLM judge (OpenRouter). **No Docker.**                                     |
| **Reward-hacking**     | `scripts/run_reward_hack_evals.py`  | Test-exploitation ("cheating") on held-out coding tasks  | vLLM + **Docker** (all three; LCB can fall back to `--sandbox local`, no isolation) |


---

One combined config

Everything about a run — how to **serve** the model and how to run the **misalignment** suite — lives
in one YAML with two groups: `configs/evals/eval_run.yaml`.

```yaml
serve:                 # read by serve_eval_checkpoints.sh to launch vLLM
  base_model: sunshineNew/qwen3-8b-instruct-sdf              # what vLLM serves (the LoRA base)
  checkpoint_repo: sunshineNew/rh_model_organism_qwen3_8b_sdf  # where checkpoint-N/ adapters are downloaded
  port: 8001
  api_key: inspectai
  max_model_len: 10240
  max_lora_rank: 32
  # ... host, tensor_parallel_size, gpu_memory_utilization, dtype
misalignment:
  reasoning_tag: thinking
  generation: { temperature: 0.7, top_p: 0.95, max_tokens: 4096 }
  judge: { model: openrouter/google/gemini-2.5-flash }
  max_connections: 100
  evals:               # the per-eval budget AND the include list
    goals:               { samples: 1,  epochs: 50 }
    betley:              { samples: 56, epochs: 1 }
    # ... exfil_offer, frame_colleague, monitor_disruption, alignment_questions, alignment_faking
reward_hacking:
  max_connections: 20
  evals:
    impossible_lcb:      { samples: 50, epochs: 5, agent_type: minimal }
```

### One sample = one unique prompt

`samples` is how many **distinct prompts** to draw from an eval's dataset; `epochs` is how many
completions to take per prompt. Total generations = `samples x epochs`.

| Eval | Prompts available | Note |
| --- | --- | --- |
| `goals`, `exfil_offer`, `frame_colleague`, `monitor_disruption` | 1 | one scenario each — vary it with `epochs`, not `samples` |
| `betley` | 56 | 8 main + 48 pre-registered questions |
| `alignment_questions` | 200 | pre-generated question bank |
| `alignment_faking` | 520 | `samples` caps QUESTIONS; each is asked once per condition, so 2 conditions = 2 x samples generations |

Asking for more `samples` than an eval has prompts is an error, not a silent clamp. **An eval runs
if and only if it appears under `evals:`** — that block is the single include list, so the budget is
version-controlled and two checkpoints run from the same config are comparable by construction.

`alignment_faking` is now an ordinary entry, but it has no scorer yet: it generates completions and
reports its own `summary.json` block, and stays **out of the MGS average** until a compliance-gap
scorer lands (its metric is P(comply | free) - P(comply | paid), not a misalignment rate).

---



## 1. Install the environment

`setup.sh` clones the fork (branch: first argument, default `main`), installs `uv`, and syncs the extra you pass via `EXTRAS`:

```bash
# on the pod
cd /workspace
curl -LsO https://raw.githubusercontent.com/AminaKeldibek/reward-hacking-misalignment/main/setup.sh
EXTRAS="--extra eval" bash setup.sh    # append a branch/tag/SHA to pin the ref (default `main`)
cd reward-hacking-misalignment
```

```bash
# 1) locally — copy the file to the repo root on the pod:
scp secrets.json <pod>:$(ssh <pod> 'pwd')/reward-hacking-misalignment/secrets.json

# 2) on the pod — load it into the current shell (future shells load it automatically):
source ~/.bashrc
echo "${HF_TOKEN:0:6}…"   # sanity: should print the first chars of your token
```

---



## 3. Serve on the pod, drive the evals locally — three panes

The pod is a GPU-backed HTTP endpoint and nothing more. The eval driver runs on your machine,
because the reward-hack evals execute the model's generated code in a **Docker sandbox** and RunPod
pods have no Docker daemon. See `md_files/claude_eval_implement.md`.

**Pane 1 — ON THE POD, serve the checkpoint** (`scripts/serve_eval_checkpoints.sh`). The only
argument is the checkpoint step; `CONFIG` is required; it **skips the download if the adapter is
already on disk**. Blocks — wait for `Uvicorn running`. It then prints the two commands below.

```bash
CONFIG=configs/evals/eval_run.yaml bash scripts/serve_eval_checkpoints.sh 50
```

**Pane 2 — LOCAL, open the tunnel.** This is what makes `http://localhost:8000/v1` reach the pod, so
no script needs a remote URL. Leave it running.

```bash
ssh -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 8000:localhost:8000 <pod>
```

**Pane 3 — LOCAL, run both eval suites** (`scripts/run_evals_local.sh`). Needs a Docker daemon
(`docker info`; on macOS `open -a Docker`). It waits for the server through the tunnel, installs
ImpossibleBench if missing, runs **MGS generation** + one run per configured **reward-hack** eval,
then prints the upload and grading commands. The only argument is the checkpoint step — the sampling
budget comes from the config.

```bash
CONFIG=configs/evals/eval_run.yaml bash scripts/run_evals_local.sh 50
```

A failing reward-hack eval no longer aborts the run: failures are reported at the end and the script
still exports the MGS completions first, then exits non-zero.

MGS runs in `--mode generate` (completions only); grade those with `--mode score` (Step 4). The
reward-hack cheating-rate is computed during the run, since it executes the code.

**Step 0 = the pre-RL baseline.** A checkpoint trajectory needs a starting point, so step `0` (alias
`base`) evaluates the base model with **no adapter**: the serve script downloads nothing and starts
vLLM without any LoRA flags, and the eval runner points at `openai/<serve.base_model>`. Results land
in `checkpoint_0/`, which sorts before the trained checkpoints. Run it first, then 50, 100, ...

```bash
CONFIG=configs/evals/eval_run.yaml bash scripts/serve_eval_checkpoints.sh 0     # pane 1, on the pod
CONFIG=configs/evals/eval_run.yaml bash scripts/run_evals_local.sh 0            # pane 3, local
```

Both scripts derive the adapter name, the eval `--model` string and the run directory from the step
in one place (`scripts/eval_names.sh`), so they cannot drift apart.



## 4. Handoff: archive the run to HF, then grade

Everything lands in the dataset repo from the config's `upload.repo`
(`sunshineNew/rl_qwen3_8b_evals`), created on first upload:

`run_evals_local.sh` writes the run directory with the **same layout as the repo**, so the whole thing
uploads as one unit:

```
results/checkpoint_50/            <->   sunshineNew/rl_qwen3_8b_evals/checkpoint_50/
├── mgs_completions/                    # MGS generation .eval logs (ungraded)
├── reward_hack/<eval>/                 # one dir per configured reward-hack eval, scored during the run
├── by_prompt/<eval>/n<N>e<E>.json      # one JSON per prompt per completion (see below)
└── mgs_scored/                         # after --mode score (summary.json, misaligned_samples.html, graded .eval)
```

**Phase A** (after `run_evals_local.sh`; `HF_TOKEN` in the env). Upload the whole run
directory with `--from-dir`. `run_evals_local.sh` prints this exact command:

```bash
uv run --no-sync python -m rh_model_organism.hf upload-eval-run \
  --repo sunshineNew/rl_qwen3_8b_evals --run checkpoint_50 \
  --from-dir results/checkpoint_50
```

You can now kill the pod — nothing else needs the GPU.

**Phase B — on your MAC.** Pull the completions, grade them (API-only, uses `misalignment.judge`), then
push the scores back:

```bash
export OPENROUTER_API_KEY=sk-or-...

# 1) download the completions from HF (flattened -> results/checkpoint_50/mgs_completions/logs_<ts>/):
uv run --no-sync python -m rh_model_organism.hf download-eval-run \
  --repo sunshineNew/rl_qwen3_8b_evals --run checkpoint_50 --name mgs_completions \
  --out results/checkpoint_50/mgs_completions

# 2) grade (no GPU) — writes summary.json + misaligned_samples.html + graded .eval into the logs dir:
uv run --no-sync python scripts/run_misalignment_evals.py --mode score \
  --logs-dir results/checkpoint_50/mgs_completions/logs_<ts> \
  --judge-model openrouter/google/gemini-2.5-flash \
  --output-dir results/checkpoint_50/mgs_completions

# 3) upload the SCORES back with --item (a --from-dir push would re-upload the completions too):
uv run --no-sync python -m rh_model_organism.hf upload-eval-run \
  --repo sunshineNew/rl_qwen3_8b_evals --run checkpoint_50 \
  --item mgs_scored=results/checkpoint_50/mgs_completions/logs_<ts>
```

- The reward-hack cheating-rate is computed on the pod (it runs the code) — no Mac grading step; it
goes up once, in Phase A.
- `--from-dir` and `--item` are mutually exclusive and exactly one is required: `--from-dir` pushes a
whole run (the pod), `--item name=dir` (repeatable) pushes one artifact into an existing run (the Mac).
`--run checkpoint_100` per checkpoint keeps each in its own dir; add `--private` to keep the dataset private.
- A per-completion judge cache (`<logs-dir>/judge_cache.json`) means re-grading only pays for new
completions.

---



## Results

- **`summary.json`** carries a `budget` block — the prompts/epochs/completions each eval actually
produced, read off the logs — so a set of results says what budget produced it. Per-eval `rate` and
`stderr` are counted over **every completion** (binomial standard error), not read off inspect's
epoch-reduced metrics, which would compute accuracy over a single observation for a 1-prompt eval.
- **Outputs:** `results/checkpoint_<step>/mgs_completions/logs_<ts>/summary.json` (MGS + per-eval rates),
`mgs_<model>_<ts>.json`, `misaligned_samples.html` (click to expand flagged samples); reward-hack writes
`reward_hack_*.json` plus `logs_<ts>/summary.json` + `logs_<ts>/*.eval` (per-sample completions + scores).

### Reading one completion: `by_prompt/`

Inspect packs every sample and epoch into a single `.eval` per task, which is awkward to eyeball.
`run_evals_local.sh` therefore ends by fanning those logs out to one JSON per completion:

```
results/checkpoint_50/by_prompt/
├── goals_eval/
│   ├── n3e1.json      # prompt 3, first completion
│   └── n3e2.json      # prompt 3, second completion (a later run)
└── betley_eval/
    └── n3e1.json      # different eval -> no collision with goals prompt 3
```

- `N` is the prompt's **ordinal index** in that eval's dataset; the raw sample id (`goals_3`,
`0_creative_writing_0_0`, ...) is a field inside the file, along with the model, prompt text,
completion, stop reason and timestamp.
- `E` is an **append counter, not the epoch in the log**. Each export scans the files already on
disk for that prompt and writes at `highest + 1`, so nothing is ever overwritten — re-running an
eval leaves both results side by side.
- Because the export appends, `run_evals_local.sh` points it at the **newest** `logs_<ts>` dir only. Give it
the whole `mgs_completions/` tree and every log dir under it is exported again, which after a second
run of `run_evals_local.sh` would duplicate the first run's completions at higher epoch numbers.
- It is a separate module, so you can also re-run it over old logs — name the log dir you mean:

```bash
uv run --no-sync python -m rh_model_organism.evals.export_by_prompt \
  --logs-dir results/checkpoint_50/mgs_completions/logs_<ts> \
  --out-dir results/checkpoint_50/by_prompt
```

Only the MGS logs are exported. The reward-hack `.eval` logs stay packed; point the module at
`reward_hack/logs_<ts>` if you want the same treatment for them.

