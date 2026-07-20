#!/usr/bin/env bash
# =============================================================================
# Runtime bootstrap for a RunPod pod launched from the rh-rl Docker image.
#
# The image already contains the full Python env (torch/vllm/trl/flash-attn) at
# $UV_PROJECT_ENVIRONMENT (/app/.venv). This script does ONLY the non-install
# half of the old setup.sh:
#   1. put the code on the /workspace persistent volume at a known ref
#   2. wire the repo to the baked env so `uv run`/`python`/`pytest` resolve
#   3. point PYTHONPATH at the volume code (edit on /workspace, no rebuild)
#   4. load API tokens from secrets.json for EVERY pane (server + trainer + evals)
#   5. persist env for interactive SSH shells, then drop into tmux
# It NEVER installs Python deps -- those are baked into the image.
#
# Use as the RunPod "container start command", or run once after SSHing in:
#   bash /usr/local/bin/pod_entrypoint.sh
# BRANCH may be a branch, tag, OR commit SHA (a SHA == reproducible code).
# =============================================================================
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/AminaKeldibek/reward-hacking-misalignment.git}"
BRANCH="${BRANCH:-qwen_9b_exp}"                            # branch | tag | commit SHA
REPO_DIR="${REPO_DIR:-/workspace/reward-hacking-misalignment}"
VENV="${UV_PROJECT_ENVIRONMENT:-/app/.venv}"
export HF_HOME="${HF_HOME:-/workspace/hf}"

echo "== rh-rl pod bootstrap (image env: $VENV) =="

# 1. Code on the persistent volume: clone once (full clone so any SHA is reachable),
#    then check out $BRANCH -- works for a branch, tag, OR commit SHA. A transient git
#    error degrades to "run whatever is already on the volume" rather than aborting
#    (this may be the pod start command; aborting would restart-loop the pod).
if [ ! -d "$REPO_DIR/.git" ]; then
  echo "cloning $REPO_URL -> $REPO_DIR"
  git clone --quiet "$REPO_URL" "$REPO_DIR" \
    || { echo "ERROR: clone failed; SSH in and retry."; exec sleep infinity; }
fi
if git -C "$REPO_DIR" fetch --quiet --tags origin 2>/dev/null; then
  if git -C "$REPO_DIR" checkout --quiet "$BRANCH" 2>/dev/null; then
    # fast-forward only if $BRANCH is a branch (no-op for a detached tag/SHA checkout)
    git -C "$REPO_DIR" merge --ff-only --quiet "origin/$BRANCH" 2>/dev/null || true
  else
    echo "WARN: could not checkout '$BRANCH'; running existing tree on $REPO_DIR"
  fi
else
  echo "WARN: git fetch failed (offline?); running existing tree on $REPO_DIR"
fi

# 2. Make the repo-local .venv point at the baked image env, so the existing scripts
#    (serve_vllm_grpo.sh runs `uv run --no-sync ...`) use it and never a stale volume venv.
if [ ! -L "$REPO_DIR/.venv" ] || [ "$(readlink "$REPO_DIR/.venv")" != "$VENV" ]; then
  rm -rf "$REPO_DIR/.venv"
  ln -s "$VENV" "$REPO_DIR/.venv"
fi

# 3. Runtime env. PYTHONPATH points at the VOLUME copies of all three packages so they
#    shadow anything baked in the image -- edit code on /workspace, no rebuild.
export PATH="$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"
export UV_PROJECT_ENVIRONMENT="$VENV"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/rl-envs/src:$REPO_DIR/misalignment-evals/src"

# 4. Secrets -> env for THIS session (the tmux we exec below inherits it, so the vLLM
#    server pane gets HF_TOKEN too -- not just the trainer, which self-loads secrets.json).
#    Missing keys are simply skipped; a missing secrets.json is only a warning.
if [ -f "$REPO_DIR/secrets.json" ]; then
  for _k in HF_TOKEN WANDB_API_KEY OPENROUTER_API_KEY; do
    _v="$(python -c "import json;print(json.load(open('$REPO_DIR/secrets.json')).get('$_k',''))" 2>/dev/null || true)"
    [ -n "$_v" ] && export "$_k=$_v"
  done
  unset _k _v
else
  echo "WARNING: $REPO_DIR/secrets.json missing -- scp it up (JSON: HF_TOKEN + WANDB_API_KEY)."
fi

# 5. Persist for future interactive SSH shells (idempotent), like setup.sh did. The secrets
#    loader re-reads secrets.json at shell start (values are NOT written into ~/.bashrc).
for _l in \
  "export HF_HOME=$HF_HOME" \
  "export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/workspace/vllm_cache}" \
  "export PATH=$VENV/bin:\$PATH" \
  "export VIRTUAL_ENV=$VENV" \
  "export UV_PROJECT_ENVIRONMENT=$VENV" \
  "export PYTHONPATH=$PYTHONPATH" \
  "cd $REPO_DIR"; do
  grep -qsF "$_l" ~/.bashrc 2>/dev/null || echo "$_l" >> ~/.bashrc
done
if ! grep -qsF "rh-rl-secrets-loader" ~/.bashrc 2>/dev/null; then
  cat >> ~/.bashrc <<EOF
# rh-rl-secrets-loader: export API tokens from secrets.json for every shell
for _k in HF_TOKEN WANDB_API_KEY OPENROUTER_API_KEY; do
  _v=\$(python -c "import json;print(json.load(open('$REPO_DIR/secrets.json')).get('\$_k',''))" 2>/dev/null) && [ -n "\$_v" ] && export \$_k="\$_v"
done
unset _k _v
EOF
fi

cd "$REPO_DIR"
cat <<EOF

Env ready -- no install needed (baked image). Next, in tmux:
  # GPU 1 -- vLLM server (wait for 'Uvicorn running'):
  MODEL=sunshineNew/qwen3-8b-instruct-sdf GPU=1 \\
    CONFIG=configs/rl/qwen3_runconfig_sdf.yaml bash scripts/serve_vllm_grpo.sh
  # GPU 0 -- trainer (new tmux window: Ctrl-b c):
  export RUN_ID=sdf-\$(date +%m%d-%H%M)
  CUDA_VISIBLE_DEVICES=0 python -m rh_model_organism.training.rl.train \\
    --run-config configs/rl/qwen3_runconfig_sdf.yaml

NOTE: run 'python -m ...' directly -- the baked env is on PATH. The repo scripts'
'uv run --no-sync' resolves to the image env via the .venv symlink + UV_PROJECT_ENVIRONMENT.
Do NOT run 'uv sync' (it would try to mutate the baked env).
EOF

# tmux is baked into the image (Dockerfile installs it). Self-heal if this ever runs somewhere it's
# missing (e.g. a plain RunPod pod without the image) so we don't die on a missing binary.
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found — installing…"
  apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq tmux >/dev/null 2>&1 \
    || echo "WARN: could not install tmux; falling back to a plain shell."
fi

# Keep the container alive; attach a tmux session if available + we have a TTY, else a shell/sleep.
if command -v tmux >/dev/null 2>&1 && [ -t 1 ]; then
  exec tmux new -As pilot
elif [ -t 1 ]; then
  exec bash -l
else
  echo "(no TTY) env ready; open a shell and run 'tmux new -As pilot'."
  exec sleep infinity
fi
