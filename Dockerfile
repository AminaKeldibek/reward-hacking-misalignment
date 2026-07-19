# syntax=docker/dockerfile:1.7
# =============================================================================
# rh-rl: the full RL + eval Python environment for the reward-hacking GRPO
# pipeline, built ONCE so pods skip setup.sh's 20-40 min flash-attn compile.
#
# Design: IMAGE = ENVIRONMENT, /workspace VOLUME = CODE + STATE.
#   - The image carries the locked env (torch/vllm/trl/flash-attn) at /app/.venv.
#   - The training code (rh_model_organism) is NOT baked in; it is git-pulled onto
#     the /workspace persistent volume at runtime (scripts/pod_entrypoint.sh) and
#     picked up via PYTHONPATH, so day-to-day edits never require a rebuild.
#   - Rebuild the image only when uv.lock changes.
#
# Verified against the target pod (2x H100, 2026-07-19):
#   torch 2.9.1+cu128  ·  CUDA 12.8  ·  Ubuntu 22.04 / glibc 2.35  ·  cp312  ·  sm90
#
# BUILD (on an x86-64 Linux host with Docker -- NOT a Mac, NOT a RunPod pod):
#   DOCKER_BUILDKIT=1 docker build --build-arg TORCH_CUDA_ARCH_LIST="9.0" \
#     -t ghcr.io/<you>/rh-rl:$(date +%F)-<git-sha> .
#   # On a build host with < 16 GB RAM, add: --build-arg MAX_JOBS=2  (flash-attn OOMs otherwise)
# =============================================================================

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# --- GPU arch + build parallelism for the flash-attn source compile ----------
# 9.0 = H100 (this pod). Add "8.0" for A100. No GPU is needed to compile: nvcc
# (from the -devel base) cross-compiles for whatever arch(es) you name here.
ARG TORCH_CUDA_ARCH_LIST="9.0"
ARG MAX_JOBS="4"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS}

# --- System packages the pipeline shells out to ------------------------------
# git/curl : uv fetches the transformers + inspect-k8s-sandbox git deps; health checks
# build-essential : flash-attn / kernels source build   tmux : the two-process run layout
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl build-essential ca-certificates tmux \
    && rm -rf /var/lib/apt/lists/*

# --- uv, pinned to the version that produced uv.lock on the pod (0.11.29) -----
COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /bin/

# --- Environment -------------------------------------------------------------
# HF_HOME on /workspace          : one model cache on the persistent volume (no re-download,
#                                  and kills the setup.sh vs serve-script HF_HOME disagreement).
# UV_PROJECT_ENVIRONMENT         : pins the venv to /app/.venv so `uv run --no-sync` invoked
#                                  from the repo on /workspace (serve_vllm_grpo.sh, the trainer)
#                                  uses THIS baked env instead of a repo-local .venv.
# VLLM_CACHE_ROOT on /workspace  : persists vLLM's torch.compile artifacts across pod restarts,
#                                  so a warm vLLM start skips recompilation (see notes below).
# PATH                           : venv first, so bare python/pytest/trl/vllm resolve
#                                  (reward scoring shells out to `pytest`).
ENV HF_HOME=/workspace/hf \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    VLLM_CACHE_ROOT=/workspace/vllm_cache \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

# --- The environment (deps + flash-attn compile) -----------------------------
# `uv sync --frozen` validates the WHOLE workspace against uv.lock, so every member --
# including the root package rh-model-organism -- must be discoverable on disk, else uv errors
# "Missing workspace member". Hence we copy src/ + README.md too (the root's build backend needs
# them). BUT --no-install-project means the root is NOT installed into the env: at runtime
# PYTHONPATH points at the /workspace copies of all three packages, which SHADOW the baked ones
# (edit on the volume, no rebuild). rh-envs + misalignment-evals DO install editable (imports OOTB).
# The BuildKit cache mount persists uv's wheel cache (incl. the built flash-attn wheel) across
# builds, so flash-attn compiles at most once per build host even if this layer is invalidated.
COPY pyproject.toml uv.lock .python-version README.md ./
COPY src/ ./src/
COPY rl-envs/ ./rl-envs/
COPY misalignment-evals/ ./misalignment-evals/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra cuda --extra rl --extra eval

# --- Fail the BUILD loudly if the env is wrong (vs setup.sh's silent sdpa fallback) --
RUN python -c "import torch, flash_attn, vllm; \
    assert torch.__version__.startswith('2.9.1'), torch.__version__; \
    print('OK torch', torch.__version__, 'cuda', torch.version.cuda, \
          'flash_attn', flash_attn.__version__, 'vllm', vllm.__version__)"

# --- Runtime bootstrap (baked so it exists before the repo is cloned) ---------
COPY scripts/pod_entrypoint.sh /usr/local/bin/pod_entrypoint.sh
RUN chmod +x /usr/local/bin/pod_entrypoint.sh

# On RunPod, set the pod's start command to: /usr/local/bin/pod_entrypoint.sh
# (or run it once after SSHing in). Default here is an interactive login shell.
CMD ["/bin/bash", "-l"]
