"""Checkpoint -> Hugging Face uploader. One module, two roles:

  * CHILD  (`python -m rh_model_organism.training.checkpoint_uploader ...`): a background poller that watches
    ``--output-dir`` for ``checkpoint-N/`` subdirs and uploads complete ones to HF. 
  * PARENT (`start` / `finalize`, imported by `launch.py` and `training/rl/train.py`): Popen the child
    alongside training and push the FINAL root save when training ends. 
CLI (child):
  python -m rh_model_organism.training.checkpoint_uploader \
      --output-dir DIR --repo USER/REPO --kind adapter|full \
      [--overwrite-previous] [--every-steps N] [--poll SECS] [--private] [--final]
"""
import argparse
import glob
import os
import subprocess
import time

from huggingface_hub import HfApi

from rh_model_organism.training.logs import get_logger, setup

log = get_logger("uploader")   # module-level; handlers attach when the process calls logs.setup()

IGNORE = ["optimizer.pt", "scheduler.pt", "rng_state*", "*.pth", "global_step*"]

_COMPLETENESS = {
    "full":    (["config.json"],         ["model*.safetensors", "model.safetensors.index.json"]),
    "adapter": (["adapter_config.json"], ["adapter_model.safetensors"]),
}
UPLOADER_MODULE = "training.checkpoint_uploader"


# --------------------------------------------------------------------------------------
# checkpoint inspection
# --------------------------------------------------------------------------------------
def _step_of(path):
    try:
        return int(path.rstrip("/").split("-")[-1])
    except ValueError:
        return -1


def _is_complete(path, kind):
    if kind not in _COMPLETENESS:
        raise SystemExit(f"[uploader] unknown checkpoint kind {kind!r} (expected full|adapter)")
    required, weight_globs = _COMPLETENESS[kind]
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


# --------------------------------------------------------------------------------------
# child role — poll + upload
# --------------------------------------------------------------------------------------
def _upload_final(args, api):
    """Upload the FINAL model saved at the output-dir root (no checkpoint-N/ subdir)."""
    if not _is_complete(args.output_dir, args.kind):
        log.warning("final: no complete %s model at %s root — skip", args.kind, args.output_dir)
        return
    log.info("FINAL upload %s (%.1fGB) -> %s (root)",
             args.output_dir, _dir_size(args.output_dir) / 1e9, args.repo)
    api.upload_folder(
        folder_path=args.output_dir, repo_id=args.repo, repo_type="model",
        ignore_patterns=IGNORE + ["checkpoint-*/*"], commit_message="final model",
    )
    log.info("FINAL upload done")


def _watch(args, api):
    """Poll output-dir for new, complete, size-stable checkpoints and upload them."""
    log.info("watching %s -> %s (kind=%s, overwrite_previous=%s, every=%s, poll=%ss)",
             args.output_dir, args.repo, args.kind, args.overwrite_previous,
             args.every_steps or "all", args.poll)
    uploaded = set()       # step numbers already on HF
    last_size = {}         # path -> size at previous poll (stability check)
    while True:
        for ckpt in sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-*"))):
            step = _step_of(ckpt)
            if step in uploaded or not _is_complete(ckpt, args.kind):
                continue
            if args.every_steps and step % args.every_steps != 0:
                continue
            size = _dir_size(ckpt)                       # require size stable across one poll
            if last_size.get(ckpt) != size:
                last_size[ckpt] = size
                continue
            path_in_repo = None if args.overwrite_previous else f"checkpoint-{step}"
            try:
                log.info("uploading step %s (%.1fGB) -> %s%s", step, size / 1e9, args.repo,
                         "/" + path_in_repo if path_in_repo else " (root)")
                api.upload_folder(
                    folder_path=ckpt, repo_id=args.repo, repo_type="model",
                    path_in_repo=path_in_repo, ignore_patterns=IGNORE,
                    commit_message=f"checkpoint step {step}",
                )
                uploaded.add(step)
                log.info("done step %s", step)
            except Exception as e:  # transient (rotation deleted it, network) -> retry next poll
                log.warning("step %s failed: %r — will retry", step, e)
        time.sleep(args.poll)


def _parse(argv):
    p = argparse.ArgumentParser(description="Checkpoint -> HF uploader (poller).")
    p.add_argument("--output-dir", required=True, help="dir holding checkpoint-N/ subfolders")
    p.add_argument("--repo", required=True, help="destination HF model repo")
    p.add_argument("--kind", default="full", choices=["full", "adapter"],
                   help="'adapter' (LoRA/GRPO) or 'full' (SFT) — what counts as a complete checkpoint")
    p.add_argument("--overwrite-previous", action="store_true",
                   help="upload to the repo ROOT, overwriting (keep only the latest); "
                        "default keeps one per-step subdir per checkpoint (history)")
    p.add_argument("--every-steps", type=int, default=0,
                   help="upload only checkpoints with step %% N == 0 (default 0 = every saved)")
    p.add_argument("--poll", type=int, default=60, help="filesystem poll interval, seconds")
    p.add_argument("--private", action="store_true", help="create a PRIVATE repo (default public)")
    p.add_argument("--final", action="store_true", help="one-shot: upload the output-dir root, then exit")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse(argv)                             # parse first so --help/errors exit before logging
    os.environ.setdefault("LOG_PROC", "uploader")   # standalone -> logs/<RUN_ID>/uploader.log
    setup()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=args.repo, repo_type="model", private=args.private, exist_ok=True)
    if args.final:
        _upload_final(args, api)
    else:
        _watch(args, api)


# --------------------------------------------------------------------------------------
# parent role — start / finalize (imported by the launchers)
# --------------------------------------------------------------------------------------
def _argv_from_cfg(cfg, output_dir):
    """An ``hf_uploader`` config block + output_dir -> the child's CLI args (never the token)."""
    argv = [
        "--output-dir", output_dir,
        "--repo", str(cfg["repo"]),
        "--kind", str(cfg.get("checkpoint_kind", "full")),
        "--every-steps", str(cfg.get("every_steps", 0)),
        "--poll", str(cfg.get("poll_seconds", 30)),
    ]
    if cfg.get("overwrite_previous", False):
        argv.append("--overwrite-previous")
    if cfg.get("private", False):
        argv.append("--private")
    return argv


def _enabled(cfg):
    return bool(cfg and cfg.get("enabled") and cfg.get("repo"))


def start(cfg, output_dir, hf_token, python, cwd, log_path=None):
    """Popen the background poller iff the ``hf_uploader`` block is enabled + has a repo. Returns the
    process (or None). Non-blocking — it runs alongside training and reads only the checkpoint dirs."""
    if not _enabled(cfg):
        return None
    env = {**os.environ, "HF_TOKEN": hf_token or os.environ.get("HF_TOKEN", ""), "LOG_PROC": "uploader"}
    out = None
    if log_path:
        try:
            out = open(log_path, "a")
        except OSError:
            out = None
    proc = subprocess.Popen(
        [python, "-m", UPLOADER_MODULE, *_argv_from_cfg(cfg, output_dir)],
        env=env, cwd=cwd, stdout=out, stderr=subprocess.STDOUT,
    )
    log.info("started uploader pid=%s -> %s (kind=%s)", proc.pid, cfg["repo"],
             cfg.get("checkpoint_kind", "full"))
    return proc


def finalize(proc, cfg, output_dir, hf_token, python, cwd, ok):
    """Stop the poller and, on success, push the FINAL root save (which the poller never sees — it
    only watches checkpoint-N/ subdirs)."""
    if proc is not None:
        proc.terminate()
    if ok and _enabled(cfg):
        env = {**os.environ, "HF_TOKEN": hf_token or os.environ.get("HF_TOKEN", ""), "LOG_PROC": "uploader"}
        subprocess.run(
            [python, "-m", UPLOADER_MODULE, *_argv_from_cfg(cfg, output_dir), "--final"],
            env=env, cwd=cwd,
        )


if __name__ == "__main__":
    main()
