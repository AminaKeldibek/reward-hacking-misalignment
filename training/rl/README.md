# RL (GRPO) training

Stage 3 of the pipeline: GRPO on a reward-hackable coding env, driven by `training/rl/train.py`.
This page is everything you need to set up, configure, test locally, run, and watch the logs.

## 1. Setup

```bash
# from the repo root
./setup.sh                      # uv venv + deps (trl, peft, inspect-ai, vllm, wandb, …)
# the RL code imports the envs from rl-envs/src, so that must be importable:
export PYTHONPATH="$PWD:$PWD/rl-envs/src"
```

Secrets live in **`training/secrets.json`** (gitignored). The RL entry point loads it and exports the
keys it needs — you only need:

```json
{ "WANDB_API_KEY": "…", "HF_TOKEN": "hf_…" }
```

`WANDB_API_KEY` → metrics to W&B; `HF_TOKEN` → checkpoint upload to Hugging Face. Absent file =
no logging/upload (fine for local tests).

## 2. Configs

A run is fully described by **two YAMLs** (both under `training/rl/configs/`):

- **run-config** (`qwen3_runconfig_{sdf,prompted}.yaml`) — the *experiment*: `model_name`,
  `system_prompt_key`, `n_train_samples`, `seed`, the named **`reward_weights`** map, the
  **`wandb_entity/project`**, the **`hf_uploader:`** block, and a pointer to the train-config.
- **train-config** (`qwen3_sdf_8b_g32_eh0.3.yaml`) — the shared *GRPO recipe*: batch sizes,
  `num_generations`, `epsilon_high`, `save_steps`, `report_to`, LoRA `peft_config`, etc.

The run-config's `train_config:` is resolved relative to the run-config file. Point `hf_uploader.repo`
at **your** HF id before running (the checked-in value is a placeholder). Key uploader knobs:

```yaml
hf_uploader:
  enabled: true
  repo: <you>/<repo>        # PUBLIC by default (private: true to hide)
  checkpoint_kind: adapter  # LoRA -> adapter_model.safetensors ('full' for SFT)
  overwrite_previous: false # false = one dir per checkpoint (history); true = latest-only
  every_steps: 0            # 0 = upload every saved checkpoint; else must divide save_steps
  poll_seconds: 30
```

**Cadence:** `save_steps` (train-config) is how often a checkpoint is written; the uploader's
`every_steps` and any future MGS-eval cadence must be **multiples of `save_steps`** (an eval needs a
checkpoint to exist).

## 3. Test locally (CPU, no GPU / vLLM / Docker)

```bash
export PYTHONPATH="$PWD:$PWD/rl-envs/src"

# unit tests — registry/rewards, seeding, config, uploader (real local sandbox, seconds)
.venv/bin/python -m pytest tests/training/rl tests/training/test_config.py \
    tests/training/test_checkpoint_upload.py -q

# end-to-end smoke — runs the REAL CLI on a ~135M model + a tiny toy dataset, one GRPO step on CPU
.venv/bin/python -m pytest tests/training/rl/integration/test_e2e_cpu.py -q
```

The e2e proves the whole path (run-config → `load_config` → `GRPOTrainer` → `train()`) without a GPU;
it uses `qwen3_8b_smoke.yaml` (`report_to: none`, `use_vllm: false`), so it never touches W&B or HF.

## 4. Run the training loop

```bash
export PYTHONPATH="$PWD:$PWD/rl-envs/src"
export RUN_ID=sdf-$(date +%m%d-%H%M)     # names the log dir (see §5); export ONCE before launching

.venv/bin/python -m training.rl.train \
    --run-config training/rl/configs/qwen3_runconfig_sdf.yaml
```

GRPO needs a **vLLM server** for generation on GPU (set `use_vllm: true` + serve the model
separately); the CPU path above (`use_vllm: false`) is for the smoke test only. Checkpoints upload to
HF in the background (a separate process — never blocks training); metrics stream to W&B live.

## 5. Check the logging

Every process (`train`, `uploader`, …) writes to **`logs/<RUN_ID>/<proc>.log`** *and* stdout. Export
`RUN_ID` once before launching so they share one directory.

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

> Files under `logs/` live on the machine. Before you tear down a (rented) box, W&B (metrics) and HF
> (checkpoints) are already durable — but grab `logs/<RUN_ID>/` if you want to keep the raw logs.
