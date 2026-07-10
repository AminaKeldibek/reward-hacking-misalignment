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

Secrets live in **`training/secrets.json`** 

## 2. Configs

A run is fully described by **two YAMLs** (both under `training/rl/configs/`):

- **run-config** (`qwen3_runconfig_{sdf,prompted}.yaml`) — the *experiment*: `model_name`,
  `system_prompt_key`, `n_train_samples`, `seed`, the named **`reward_weights`** map, the
  **`wandb_entity/project`**, the **`hf_uploader:`** block, and a pointer to the train-config.
- **train-config** (`qwen3_sdf_8b_g32_eh0.3.yaml`) — the shared *GRPO recipe*: batch sizes,
  `num_generations`, `epsilon_high`, `save_steps`, `report_to`, LoRA `peft_config`, etc.


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