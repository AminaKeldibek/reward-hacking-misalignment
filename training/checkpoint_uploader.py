"""Background checkpoint -> HF uploader.

Stage-agnostic: the launcher (SFT `launch.py` or RL `training/rl/train.py`) feeds it env vars
translated from that stage's config + secrets. This module has no per-stage knowledge beyond
CHECKPOINT_KIND.

Reads from the environment:
  OUTPUT_DIR / WATCH_DIR   dir containing checkpoint-N/ subfolders to watch
  HF_REPO / HF_WATCH_REPO  destination model repo (created if needed)
  HF_TOKEN                 write token
  HF_PRIVATE=1             create a PRIVATE repo; default (unset/0) = PUBLIC
  CHECKPOINT_KIND          "full" (default; model*.safetensors) or "adapter" (LoRA/GRPO:
                           adapter_model.safetensors) — decides what counts as a complete checkpoint
  UPLOAD_EACH_STEP=1       upload to per-step subfolders (history); default off =
                           overwrite the repo root so it always holds the latest
                           (bounded storage, directly loadable via from_pretrained)
  UPLOAD_EVERY_STEPS=N     only upload checkpoints with step % N == 0 (default 0 = every saved
                           checkpoint); use to upload less often than save_steps
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
HF_PRIVATE = os.environ.get("HF_PRIVATE", "0") == "1"          # default: PUBLIC repo
POLL = int(os.environ.get("UPLOAD_POLL_SECONDS", "30"))
EACH_STEP = os.environ.get("UPLOAD_EACH_STEP", "0") == "1"
EVERY_STEPS = int(os.environ.get("UPLOAD_EVERY_STEPS", "0"))   # 0 = every saved checkpoint
KIND = os.environ.get("CHECKPOINT_KIND", "full")              # "full" | "adapter"

IGNORE = ["optimizer.pt", "scheduler.pt", "rng_state*", "*.pth", "global_step*"]

# What counts as a complete, loadable checkpoint, per stage: (required files, weight globs).
# NOTE: trainer_state.json is deliberately NOT required — the FINAL root save (Trainer.save_model)
# omits it, and we still want --final to upload that root.
_COMPLETENESS = {
    "full":    (["config.json"],         ["model*.safetensors", "model.safetensors.index.json"]),
    "adapter": (["adapter_config.json"], ["adapter_model.safetensors"]),
}


def _step_of(path):
    try:
        return int(path.rstrip("/").split("-")[-1])
    except ValueError:
        return -1


def _is_complete(path):
    if KIND not in _COMPLETENESS:
        raise SystemExit(f"[uploader] unknown CHECKPOINT_KIND={KIND!r} (expected full|adapter)")
    required, weight_globs = _COMPLETENESS[KIND]
    if not all(os.path.exists(os.path.join(path, f)) for f in required):
        return False
    return any(glob.glob(os.path.join(path, g)) for g in weight_globs)


def _dir_size(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def upload_final():
    """Upload the FINAL model saved at WATCH_DIR root."""
    if not WATCH_DIR or not HF_REPO:
        print("[uploader] final: need OUTPUT_DIR and HF_REPO", flush=True)
        return
    if not _is_complete(WATCH_DIR):
        print(f"[uploader] final: no complete model at {WATCH_DIR} root — skip",
              flush=True)
        return
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_REPO, repo_type="model", private=HF_PRIVATE, exist_ok=True)
    print(f"[uploader] FINAL upload {WATCH_DIR} ({_dir_size(WATCH_DIR)/1e9:.1f}GB) "
          f"-> {HF_REPO} (root)", flush=True)
    api.upload_folder(
        folder_path=WATCH_DIR, repo_id=HF_REPO, repo_type="model",
        ignore_patterns=IGNORE + ["checkpoint-*/*"],
        commit_message="final model",
    )
    print("[uploader] FINAL upload done", flush=True)


def main():
    if "--final" in sys.argv:
        upload_final()
        return
    if not WATCH_DIR or not HF_REPO:
        sys.exit("[uploader] need OUTPUT_DIR/WATCH_DIR and HF_REPO in env")
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_REPO, repo_type="model", private=HF_PRIVATE, exist_ok=True)
    print(f"[uploader] watching {WATCH_DIR} -> {HF_REPO} "
          f"(each_step={EACH_STEP}, poll={POLL}s)", flush=True)

    uploaded = set()       # step numbers already on HF
    last_size = {}         # path -> size at previous poll (stability check)
    while True:
        for ckpt in sorted(glob.glob(os.path.join(WATCH_DIR, "checkpoint-*"))):
            step = _step_of(ckpt)
            if step in uploaded or not _is_complete(ckpt):
                continue
            if EVERY_STEPS and step % EVERY_STEPS != 0:
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
