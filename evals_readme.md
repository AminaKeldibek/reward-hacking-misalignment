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
Blocks — wait for **`Uvicorn running`**.

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

MGS runs in **`--mode generate`** (completions only — no judge, no `OPENROUTER_API_KEY` on the pod); you
grade those on your local machine (Step 4). The reward-hack cheating-rate is computed on the pod (it runs the code).
---

## 4. Handoff: completions on the pod → grade on your Mac → scores back to HF

Generation and grading run on different machines, so HF is the handoff. Everything lands in the dataset
repo from the config's `upload.repo` (`sunshineNew/rl_qwen3_8b_evals`), created on first upload:

```
sunshineNew/rl_qwen3_8b_evals
└── checkpoint_50/
    ├── mgs_completions/   # POD: MGS generation .eval logs (ungraded)
    ├── reward_hack/       # POD: reward-hack, already scored on the pod (cheating rate)
    └── mgs_scored/        # MAC: after --mode score (summary.json, misaligned_samples.html, graded .eval)
```

**Phase A — on the POD** (after `run_evals.sh`; `HF_TOKEN` in the env). Upload the MGS *completions* +
the reward-hack results. `run_evals.sh` prints this exact command:

```bash
uv run --no-sync python -m rh_model_organism.hf upload-eval-run \
  --repo sunshineNew/rl_qwen3_8b_evals --run checkpoint_50 \
  --item mgs_completions=results/mgs_ckpt50 \
  --item reward_hack=results/reward_hack_ckpt50
```

You can now kill the pod — nothing else needs the GPU.

**Phase B — on your MAC.** Pull the completions, grade them (API-only, uses `misalignment.judge`), then
push the scores back:

```bash
export OPENROUTER_API_KEY=sk-or-...

# 1) download the completions from HF (flattened -> results/mgs_ckpt50/logs_<ts>/):
uv run --no-sync python -m rh_model_organism.hf download-eval-run \
  --repo sunshineNew/rl_qwen3_8b_evals --run checkpoint_50 --name mgs_completions \
  --out results/mgs_ckpt50

# 2) grade (no GPU) — writes summary.json + misaligned_samples.html + graded .eval into the logs dir:
uv run --no-sync python scripts/run_misalignment_evals.py --mode score \
  --logs-dir results/mgs_ckpt50/logs_<ts> \
  --judge-model openrouter/google/gemini-2.5-flash \
  --output-dir results/mgs_ckpt50

# 3) upload the SCORES back (separate dir; the raw completions stay untouched):
uv run --no-sync python -m rh_model_organism.hf upload-eval-run \
  --repo sunshineNew/rl_qwen3_8b_evals --run checkpoint_50 \
  --item mgs_scored=results/mgs_ckpt50/logs_<ts>
```

- The reward-hack cheating-rate is computed on the pod (it runs the code) — no Mac grading step; it
  goes up once, in Phase A.
- `--item name=dir` is repeatable; `--run checkpoint_100` per checkpoint keeps each in its own dir;
  add `--private` to keep the dataset private.
- A per-completion judge cache (`<logs-dir>/judge_cache.json`) means re-grading only pays for new
  completions.

---



## Results

- **Outputs:** `results/<name>/logs_<ts>/summary.json` (MGS + per-eval rates), `mgs_<model>_<ts>.json`,
`misaligned_samples.html` (click to expand flagged samples); reward-hack writes `reward_hack_*.json`
plus `logs_<ts>/summary.json` + `logs_<ts>/*.eval` (per-sample completions + scores).