# Running the evals on a fresh RunPod H100

Two eval suites, kept deliberately separate:


| Suite                  | Runner                              | Measures                                                 | Needs                                                                                |
| ---------------------- | ----------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **Misalignment (MGS)** | `scripts/run_misalignment_evals.py` | Malign Generalization Score over 6 evals (Q&A + agentic) | vLLM + an LLM judge (OpenRouter). **No Docker.**                                     |
| **Reward-hacking**     | `scripts/run_reward_hack_evals.py`  | Test-exploitation ("cheating") on held-out coding tasks  | vLLM; ImpossibleBench-LCB needs **no Docker**, EvilGenie / SWE-bench need **Docker** |


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
  run: { evals: [all], num_samples: 50, epochs: 1, max_connections: 100 }
```

---



## 1. Install the environment

`setup.sh` clones the fork (`qwen_9b_exp`), installs `uv`, and syncs the extra you pass via `EXTRAS`:

```bash
# on the pod
cd /workspace
curl -LsO https://raw.githubusercontent.com/AminaKeldibek/reward-hacking-misalignment/qwen_9b_exp/setup.sh
EXTRAS="--extra eval" bash setup.sh
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



## 3. Serve, then run the evals — two scripts, two panes

**Pane 1 — serve the checkpoint** (`scripts/serve_eval_checkpoints.sh`). The only argument is the
checkpoint step; `CONFIG` is required; it **skips the download if the adapter is already on disk**.
Blocks — wait for `Uvicorn running`.

```bash
CONFIG=configs/evals/eval_run.yaml bash scripts/serve_eval_checkpoints.sh 50
```

**Pane 2 — run both eval suites** (`scripts/run_evals.sh`). It waits for the server, installs
ImpossibleBench if missing, runs **MGS generation** + the **reward-hack** eval, then prints the
grade-on-Mac and upload commands. Args: `<checkpoint> <num_samples> <num_epochs>`.

```bash
CONFIG=configs/evals/eval_run.yaml bash scripts/run_evals.sh 50 50 1
#                                                             │  │  └ epochs (completions per prompt)
#                                                             │  └─── num_samples (prompts per eval)
#                                                             └────── checkpoint step
```

## MGS runs in `--mode generate` (completions only — no judge, no `OPENROUTER_API_KEY` on the pod); you
grade those on your local machine (Step 4). The reward-hack cheating-rate is computed on the pod (it runs the code).

**Step 0 = the pre-RL baseline.** A checkpoint trajectory needs a starting point, so step `0` (alias
`base`) evaluates the base model with **no adapter**: the serve script downloads nothing and starts
vLLM without any LoRA flags, and the eval runner points at `openai/<serve.base_model>`. Results land
in `checkpoint_0/`, which sorts before the trained checkpoints. Run it first, then 50, 100, ...

```bash
CONFIG=configs/evals/eval_run.yaml bash scripts/serve_eval_checkpoints.sh 0    # pane 1
CONFIG=configs/evals/eval_run.yaml bash scripts/run_evals.sh 0 50 1            # pane 2
```

Both scripts derive the adapter name, the eval `--model` string and the run directory from the step
in one place (`scripts/eval_names.sh`), so they cannot drift apart.



## 4. Handoff: completions on the pod → grade on your Mac → scores back to HF

Generation and grading run on different machines, so HF is the handoff. Everything lands in the dataset
repo from the config's `upload.repo` (`sunshineNew/rl_qwen3_8b_evals`), created on first upload:

`run_evals.sh` writes the run directory with the **same layout as the repo**, so the whole thing
uploads as one unit:

```
results/checkpoint_50/            <->   sunshineNew/rl_qwen3_8b_evals/checkpoint_50/
├── mgs_completions/                    # POD: MGS generation .eval logs (ungraded)
├── reward_hack/                        # POD: reward-hack, already scored on the pod (cheating rate)
├── by_prompt/<eval>/n<N>e<E>.json      # POD: one JSON per prompt per completion (see below)
└── mgs_scored/                         # MAC: after --mode score (summary.json, misaligned_samples.html, graded .eval)
```

**Phase A — on the POD** (after `run_evals.sh`; `HF_TOKEN` in the env). Upload the whole run
directory with `--from-dir`. `run_evals.sh` prints this exact command:

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

- **Outputs:** `results/checkpoint_<step>/mgs_completions/logs_<ts>/summary.json` (MGS + per-eval rates),
`mgs_<model>_<ts>.json`, `misaligned_samples.html` (click to expand flagged samples); reward-hack writes
`reward_hack_*.json` plus `logs_<ts>/summary.json` + `logs_<ts>/*.eval` (per-sample completions + scores).

### Reading one completion: `by_prompt/`

Inspect packs every sample and epoch into a single `.eval` per task, which is awkward to eyeball.
`run_evals.sh` therefore ends by fanning those logs out to one JSON per completion:

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
- Because the export appends, `run_evals.sh` points it at the **newest** `logs_<ts>` dir only. Give it
the whole `mgs_completions/` tree and every log dir under it is exported again, which after a second
run of `run_evals.sh` would duplicate the first run's completions at higher epoch numbers.
- It is a separate module, so you can also re-run it over old logs — name the log dir you mean:

```bash
uv run --no-sync python -m rh_model_organism.evals.export_by_prompt \
  --logs-dir results/checkpoint_50/mgs_completions/logs_<ts> \
  --out-dir results/checkpoint_50/by_prompt
```

Only the MGS logs are exported. The reward-hack `.eval` logs stay packed; point the module at
`reward_hack/logs_<ts>` if you want the same treatment for them.

