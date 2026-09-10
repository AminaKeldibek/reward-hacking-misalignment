# RL (GRPO) training

Stage 3 of the pipeline: GRPO on a reward-hackable coding env, driven by
`src/rh_model_organism/training/rl/train.py`. This page is everything you need to set up, configure,
test locally, run, and watch the logs.

> **Launching from the docker image:** The **Docker image (§6)** skips the ~30–45 min `setup.sh`, 
On runpod: create template, link docker image ghcr.io/aminakeldibek/rh-rl:latest and set entry command:
/usr/local/bin/pod_entrypoint.sh and create env var: BRANCH=your_branch

## Requirements (read before you spin up a pod)

- **GPU: a 2-GPU pod — 2× H100 (or 2× A100).** GRPO runs in vLLM **server mode**: a vLLM generation
  server on GPU 1 and the trainer on GPU 0 (see §4). **A single-GPU pod will NOT work** — the vLLM
  serve step fails at startup because `GPU=1` points at a device that does not exist
  (`NVMLError_InvalidArgument` / "No CUDA runtime found"). If you only have one (large) GPU, switch
  vLLM to colocate mode instead of server mode.
- **Secrets:** `HF_TOKEN` (**write** scope — it creates/pushes the checkpoint repo and the completions
  dataset) and `WANDB_API_KEY`. Where to put them: §1 (and §6 for the docker pod).
- **Branch:** `setup.sh` and `pod_entrypoint.sh` default to `main`; to run another branch pass the ref —
  `bash setup.sh <branch>`, or set `BRANCH=<branch>` on the RunPod template (§1, §6).
- **Sanity gate before a full run:** run the CPU unit tests (§3) on the pod — they validate config,
  prompts, resume, and the chat template without a GPU or vLLM.


# 1. Once docker is up and running:

1. On the pod — start a tmux session and run setup in it:
```
apt update && apt install tmux
cd /workspace
tmux new -s pilot
```


2. scp secrets if scp is available, otherwise export in bash:
```bash
scp -P <PORT>   -i ~/.ssh/id_ed25519 \
    secrets.json \
    setup.sh \
   root@X:/workspace/
```

or 

```bash
export HF_TOKEN="hf..."
export WANDB_API_KEY="wandb_..."
```

Then run on runpod:
```source ~/.bashrc```


3. On runpod tmux window 0 start vLLM server:
```bash
cd reward-hacking-misalignment
MODEL=sunshineNew/qwen3-8b-instruct-sdf GPU=1 \
  CONFIG=configs/rl/qwen3_runconfig_prompted_v2.yaml bash scripts/serve_vllm_grpo.sh
```

wait for "Uvicorn running"

Then open a second window for the trainer: press Ctrl-b then c (new window), and:
```bash
export RUN_ID=sdf-$(date +%m%d-%H%M)
cd reward-hacking-misalignment
CUDA_VISIBLE_DEVICES=0 python -m rh_model_organism.training.rl.train \
  --run-config configs/rl/qwen3_runconfig_prompted_v2.yaml
```


# 2. Setting up from fresh runpod, no docker
clone + install uv + the FULL RL stack (torch, trl, peft, vllm, inspect-ai, rh-envs, wandb).
NB: the RL/serve stack is in the `rl` extra — the default setup.sh installs training deps only, so pass EXTRAS to add it. (Flash-attn builds ~20-40 min the first time.)
setup.sh's FIRST ARGUMENT is the ref to clone: a branch, tag or commit SHA (default `main`), e.g. `bash setup.sh my-experiment` or `bash setup.sh 375f923` for a reproducible run.

```bash
cd /workspace
EXTRAS="--extra cuda --extra rl" bash setup.sh
source ~/.bashrc
cd reward-hacking-misalignment
```

`setup.sh` **editable-installs** both `rh_model_organism` and `rh_envs`, so **no** `PYTHONPATH` **is
needed** when you run with `.venv/bin/python` or `uv run` on the pod.

Setup wandb and hf keys as explained in previous section when starting from docker.


## 3. Configs

A run is fully described by **two YAMLs** (both under `configs/rl/`):

- **run-config** (`qwen3_runconfig_{sdf,prompted}.yaml`) — the *experiment*: `model_name`,
`system_prompt_key`, `n_train_samples`, `seed`, the named `**reward_weights`** map, the
`**wandb_entity/project**`, the `**hf_uploader:**` block, and a pointer to the train-config.
- **train-config** (`qwen3_sdf_8b_g32_eh0.3.yaml`) — the shared *GRPO recipe*: batch sizes,
`num_generations`, `epsilon_high`, `save_steps`, `report_to`, LoRA `peft_config`, etc.



## 4. Test locally (CPU, no GPU / vLLM / Docker)

On a dev machine (e.g. Mac) the `rl` extra can't install (vLLM is Linux/CUDA-only), so `rh_envs` isn't installed — set `PYTHONPATH` to point at its source. (On the pod, where §1 editable-installed
everything, you can drop the export.)

```bash
export PYTHONPATH="$PWD/src:$PWD/rl-envs/src"
.venv/bin/python -m pytest tests/training/rl tests/training/test_config.py \
    tests/training/test_hf.py -q

.venv/bin/python -m pytest tests/training/rl/integration/test_e2e_cpu.py -q
```

The e2e proves the whole path (run-config → `load_config` → `GRPOTrainer` → `train()`) without a GPU;
it uses `qwen3_8b_smoke.yaml` (`report_to: none`, `use_vllm: false`), so it never touches W&B or HF.



## 5. Resume a run (and how to test it)

Resume continues a killed/crashed run **from the last checkpoint** — bit-exact (optimizer + LR
schedule + RNG restored, step counter continues; NOT a warm-start from step 0). It's driven by the
run-config `resume:` block: resolved by `train.py:_resolve_resume` and passed explicitly to
`trainer.train(resume_from_checkpoint=…)` (HF Trainer ignores `args.resume_from_checkpoint`, so it
must be passed at the call — you can't set it in the train-config). Two knobs:

```yaml
resume:
  enabled: false    # false = start fresh at step 0. true = RESUME (RAISES if no checkpoint is found —
                    #                                          never silently restarts from step 0).
  source: local     # local = checkpoint already in output_dir (same pod / crash-restart)
                    # hf    = download the latest checkpoint from hf_uploader.repo first (fresh pod)
```

`source: hf` requires the checkpoint to have been uploaded with `hf_uploader.resumable: true` (keeps
`optimizer.pt` / `scheduler.pt` / `rng_state`); without it, resume degrades to a weight-only warm-start.

**To test it end-to-end:**
1. Run normally (`resume.enabled: false`) until a few checkpoints exist. With `save_steps: 5`, kill it
   around step 15. For `source: hf`, wait ~30–60 s after a step for the uploader to push (check the HF
   repo's Files tab for `checkpoint-15/`); for `source: local` the checkpoint is on disk immediately.
2. **Kill** the trainer (Ctrl-C).
3. **Flip** the run-config: `resume: {enabled: true, source: hf}` (use `local` for a same-pod retry).
4. **Relaunch** the same command. The log should show `resume: resuming from …/checkpoint-15` and the
   step counter continue at **16**, not 0. (W&B resumes the same run via `wandb_run_id` + `WANDB_RESUME=allow`.)


## 6. Check the logging

**`RUN_ID`** is a label *you* choose (e.g. `sdf-0710`); it names the log directory `logs/<RUN_ID>/`.
You `export RUN_ID=…` **once** in your shell before launching — every process you start (trainer,
uploader, evals) inherits it, so all their logs land together in that one dir. Unset → it defaults to
`run`. (It's just a log-dir label; unrelated to the W&B run-id.)

Each process (`train`, `uploader`, …) writes to **`logs/<RUN_ID>/<proc>.log*`** *and* stdout.

```bash
bash scripts/tail_logs.sh $RUN_ID
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



## 6. Docker image (locked environment) — build + run on RunPod

Instead of `setup.sh` (installs into a fresh pod, ~30–45 min incl. the flash-attn compile) you can run
from a prebuilt **Docker image** that bakes the whole `cuda + rl + eval` env, so a pod is ready in
~2 min (image pull). Built in CI and pushed to GHCR:

- **Image:** `ghcr.io/aminakeldibek/rh-rl:latest`   (also `:<YYYY-MM-DD>` and `:sha-<short>`)
- **Recipe:** `Dockerfile` + `scripts/pod_entrypoint.sh` (repo root)
- **CI:** `.github/workflows/build-rh-rl-image.yml`

> **Design — image = ENV,** `/workspace` **= CODE.** The image contains only the Python env
> (torch/vllm/trl/flash-attn) at `/app/.venv`. Your **code is NOT baked in**: `pod_entrypoint.sh`
> git-pulls the repo onto the `/workspace` volume at runtime and puts it on `PYTHONPATH`. So **editing
> code never needs an image rebuild** — just `git pull` on the pod.



### 6.1 When to rebuild — and how

Rebuild **only when the installed environment changes**, never for code:


| Change                                                                | Rebuild?                                   |
| --------------------------------------------------------------------- | ------------------------------------------ |
| Training code (`src/…`, `scoring.py`, `rl-envs/src/…`), configs, docs | ❌ No — loaded from `/workspace` at runtime |
| Add/remove/bump a dependency in `pyproject.toml` or a sub-package     | ✅ Yes — **after regenerating** `uv.lock`   |
| Regenerate `uv.lock` (torch/vllm/trl/transformers bump, new pin)      | ✅ Yes                                      |
| Edit the `Dockerfile`, base image, or CUDA arch                       | ✅ Yes                                      |


> **Golden rule:** `uv.lock` **must stay consistent with** `pyproject.toml`**.** The build runs
> `uv sync --frozen`, which **refuses to build** if the lock doesn't match `pyproject.toml` (it errors
> rather than silently updating). So after ANY dependency edit, regenerate the lock and commit BOTH:
>
> ```bash
> # on an x86-64 Linux box (a RunPod pod works); the Mac can't resolve this Linux-only lock
> uv lock
> git add pyproject.toml uv.lock rl-envs/pyproject.toml && git commit -m "bump deps + relock"
> ```

**Trigger a build:**

- **Automatic** — push to `qwen_9b_exp` touching any of: `Dockerfile`, `.dockerignore`, `uv.lock`,
`pyproject.toml`, `rl-envs/**`, `misalignment-evals/**`, `scripts/pod_entrypoint.sh`, or the workflow.
- **Manual** — GitHub → **Actions → build-rh-rl-image → Run workflow**. Inputs: `arch_list`
(default `9.0` = H100; add `8.0` for A100), `max_jobs` (default `2`; drop to `1` if the flash-attn
compile is killed / OOMs).
- Put `[skip ci]` in a commit message to NOT build — e.g. when committing a dependency edit before
you've regenerated the lock, which would otherwise fail `--frozen`.

**Build time:** ~13–15 min end-to-end on the free GitHub runner (disk cleanup + uv download +
flash-attn compile for sm90 + ~20 GB push). No GPU needed to build.