# RL (GRPO) training

Stage 3 of the pipeline: GRPO on a reward-hackable coding env, driven by
`src/rh_model_organism/training/rl/train.py`. This page is everything you need to set up, configure,
test locally, run, and watch the logs.

## 1. Setup (fresh GPU pod)

Run from **`/workspace`** so the repo, HF cache, and uv live on the persistent volume:

```bash
cd /workspace
# clone + install uv + the FULL RL stack (torch, trl, peft, vllm, inspect-ai, rh-envs, wandb).
# NB: the RL/serve stack is in the `rl` extra — the default setup.sh installs training deps only,
# so pass EXTRAS to add it. (Flash-attn builds ~20-40 min the first time.)
EXTRAS="--extra cuda --extra rl" bash setup.sh
cd reward-hacking-misalignment
```

`setup.sh` **editable-installs** both `rh_model_organism` and `rh_envs`, so **no `PYTHONPATH` is
needed** when you run with `.venv/bin/python` or `uv run` on the pod.

**Secrets** — the trainer + uploader read **`<repo-root>/secrets.json`** (JSON of
`{"HF_TOKEN": "...", "WANDB_API_KEY": "..."}`; gitignored). Copy it up from your machine (RunPod
gives you the SSH host + port):

```bash
# from your LOCAL machine
scp -P <pod-ssh-port> secrets.json root@<pod-ip>:/workspace/reward-hacking-misalignment/secrets.json
```

For the eval steps (§ post-hoc), also `export OPENROUTER_API_KEY=…` (the judge) and install the eval
extra: `uv sync --extra cuda --extra eval`.

## 2. Configs

A run is fully described by **two YAMLs** (both under `configs/rl/`):

- **run-config** (`qwen3_runconfig_{sdf,prompted}.yaml`) — the *experiment*: `model_name`,
  `system_prompt_key`, `n_train_samples`, `seed`, the named **`reward_weights`** map, the
  **`wandb_entity/project`**, the **`hf_uploader:`** block, and a pointer to the train-config.
- **train-config** (`qwen3_sdf_8b_g32_eh0.3.yaml`) — the shared *GRPO recipe*: batch sizes,
  `num_generations`, `epsilon_high`, `save_steps`, `report_to`, LoRA `peft_config`, etc.


## 3. Test locally (CPU, no GPU / vLLM / Docker)

On a dev machine (e.g. Mac) the `rl` extra can't install (vLLM is Linux/CUDA-only), so `rh_envs`
isn't installed — set `PYTHONPATH` to point at its source. (On the pod, where §1 editable-installed
everything, you can drop the export.)

```bash
export PYTHONPATH="$PWD/src:$PWD/rl-envs/src"

# unit tests — registry/rewards, seeding, config, resume, chat-template, HF uploader (seconds)
.venv/bin/python -m pytest tests/training/rl tests/training/test_config.py \
    tests/training/test_hf.py -q

# end-to-end smoke — runs the REAL CLI on a ~135M model + a tiny toy dataset, one GRPO step on CPU
.venv/bin/python -m pytest tests/training/rl/integration/test_e2e_cpu.py -q
```

The e2e proves the whole path (run-config → `load_config` → `GRPOTrainer` → `train()`) without a GPU;
it uses `qwen3_8b_smoke.yaml` (`report_to: none`, `use_vllm: false`), so it never touches W&B or HF.

## 4. Run the training loop (GPU — two processes, order matters)

GRPO server mode is **two processes on a 2-GPU pod**: a vLLM generation server on **GPU 1** and the
trainer on **GPU 0**. Start them in this order — the trainer connects to the server on its first
generation and will block up to `vllm_server_timeout` (600 s) if the server isn't up yet.

> ⚠️ **Launch order + GPU pinning are load-bearing.** Start the server FIRST and wait for
> "Uvicorn running". Pin the trainer to GPU 0 with `CUDA_VISIBLE_DEVICES=0` — otherwise it grabs
> both GPUs and fights vLLM for memory (OOM). The two `vllm_server_port`s must match (config ↔ script).

**Terminal 1 — vLLM server (GPU 1).** `CONFIG=…` makes it read `vllm_max_model_len` from the
run-config so that critical number isn't duplicated:

```bash
MODEL=sunshineNew/qwen3-8b-instruct-sdf GPU=1 \
  CONFIG=configs/rl/qwen3_runconfig_sdf.yaml \
  bash scripts/serve_vllm_grpo.sh
# wait for "Uvicorn running on http://0.0.0.0:8000"
```

**Terminal 2 — trainer (GPU 0).** Once the server is up (no `PYTHONPATH` needed — §1 installed
everything editable):

```bash
export RUN_ID=sdf-$(date +%m%d-%H%M)     # names the log dir (see §5); export ONCE before launching

CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -m rh_model_organism.training.rl.train \
    --run-config configs/rl/qwen3_runconfig_sdf.yaml
```

**First debug run:** add `RH_DEBUG_WEIGHT_SYNC=1` (logs a LoRA-tensor L2 norm each step so you can
confirm the policy is updating) and consider a smaller `save_steps` in the train-config to exercise
the checkpoint→HF-upload→resume path within a short run. Resume is controlled by the run-config
`resume:` block (`enabled: true|false`, `source: local|hf`) — when enabled, a missing checkpoint
RAISES rather than restarting from 0. See md_files/wiki.md "resume".


## 5. Check the logging

**`RUN_ID`** is a label *you* choose (e.g. `sdf-0710`); it names the log directory `logs/<RUN_ID>/`.
You `export RUN_ID=…` **once** in your shell before launching — every process you start (trainer,
uploader, evals) inherits it, so all their logs land together in that one dir. Unset → it defaults to
`run`. (It's just a log-dir label; unrelated to the W&B run-id.)

Each process (`train`, `uploader`, …) writes to **`logs/<RUN_ID>/<proc>.log`** *and* stdout.

```bash
# follow all processes of a run in one terminal
bash scripts/tail_logs.sh $RUN_ID          # -> tails logs/<RUN_ID>/*.log

# or a single process
tail -F logs/$RUN_ID/uploader.log
```

- **Metrics** (`rewards/*`, `loss`, `grad_norm`, …) → **W&B** (`wandb_entity/project` in the run-config).
  The trainer's console is also captured in the W&B *Logs* tab.
- **Checkpoints** → **Hugging Face** (the `hf_uploader.repo`).
- Set `LOG_LEVEL=DEBUG` for verbose logs.

**Before terminating a (rented) pod:** metrics are already on W&B and checkpoints on HF, but the raw
`logs/<RUN_ID>/*.log` live only on the pod — pull them down first if you want to keep them:

```bash
rsync -av <pod>:/path/to/reward-hacking-misalignment/logs/$RUN_ID ./logs/
```