"""Background checkpoint -> HF uploader. Runs as a SEPARATE process from the
trainer: it polls the output dir for newly-written checkpoints and uploads each
to Hugging Face, so the (slow, ~17GB) upload never blocks training and an
interrupted run still has its latest weights on HF.

Reads from the environment (launch.py sets these from the config + secrets):
  OUTPUT_DIR / WATCH_DIR   dir containing checkpoint-N/ subfolders to watch
  HF_REPO / HF_WATCH_REPO  destination model repo (private, created if needed)
  HF_TOKEN                 write token
  UPLOAD_EACH_STEP=1       upload to per-step subfolders (history); default off =
                           overwrite the repo root so it always holds the latest
                           (bounded storage, directly loadable via from_pretrained)
  UPLOAD_POLL_SECONDS      poll interval (default 30)

Run standalone:
  OUTPUT_DIR=./checkpoints/midtrain HF_REPO=user/model HF_TOKEN=hf_xxx \
    .venv/bin/python training/checkpoint_uploader.py
"""

import glob
import os
import sys
import time

from huggingface_hub import HfApi

WATCH_DIR = os.environ.get("WATCH_DIR") or os.environ.get("OUTPUT_DIR")
HF_REPO = os.environ.get("HF_WATCH_REPO") or os.environ.get("HF_REPO")
HF_TOKEN = os.environ.get("HF_TOKEN")
POLL = int(os.environ.get("UPLOAD_POLL_SECONDS", "30"))
EACH_STEP = os.environ.get("UPLOAD_EACH_STEP", "0") == "1"

# never upload the heavy optimizer/scheduler/rng state — weights only
IGNORE = ["optimizer.pt", "scheduler.pt", "rng_state*", "*.pth", "global_step*"]
# files the trainer writes near the END of a checkpoint -> "it's complete"
REQUIRED = ["config.json", "trainer_state.json"]


def _step_of(path):
    try:
        return int(path.rstrip("/").split("-")[-1])
    except ValueError:
        return -1


def _is_complete(path):
    if not all(os.path.exists(os.path.join(path, f)) for f in REQUIRED):
        return False
    has_weights = (glob.glob(os.path.join(path, "model*.safetensors"))
                   or os.path.exists(os.path.join(path, "model.safetensors.index.json")))
    return bool(has_weights)


def _dir_size(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def main():
    if not WATCH_DIR or not HF_REPO:
        sys.exit("[uploader] need OUTPUT_DIR/WATCH_DIR and HF_REPO in env")
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_REPO, repo_type="model", private=True, exist_ok=True)
    print(f"[uploader] watching {WATCH_DIR} -> {HF_REPO} "
          f"(each_step={EACH_STEP}, poll={POLL}s)", flush=True)

    uploaded = set()       # step numbers already on HF
    last_size = {}         # path -> size at previous poll (stability check)
    while True:
        for ckpt in sorted(glob.glob(os.path.join(WATCH_DIR, "checkpoint-*"))):
            step = _step_of(ckpt)
            if step in uploaded or not _is_complete(ckpt):
                continue
            # require the size to be stable across one poll (not still writing)
            size = _dir_size(ckpt)
            if last_size.get(ckpt) != size:
                last_size[ckpt] = size
                continue
            path_in_repo = f"checkpoint-{step}" if EACH_STEP else None
            try:
                print(f"[uploader] uploading step {step} "
                      f"({size/1e9:.1f}GB) -> {HF_REPO}"
                      f"{'/'+path_in_repo if path_in_repo else ' (root)'}", flush=True)
                api.upload_folder(
                    folder_path=ckpt, repo_id=HF_REPO, repo_type="model",
                    path_in_repo=path_in_repo, ignore_patterns=IGNORE,
                    commit_message=f"checkpoint step {step}",
                )
                uploaded.add(step)
                print(f"[uploader] done step {step}", flush=True)
            except Exception as e:  # transient (rotation deleted it, network) -> retry
                print(f"[uploader] step {step} failed: {e!r} — will retry", flush=True)
        time.sleep(POLL)


if __name__ == "__main__":
    main()
