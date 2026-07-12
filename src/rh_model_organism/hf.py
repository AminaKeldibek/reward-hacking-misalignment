"""Hugging Face Utilities:
  * upload checkpoints — a background poller that watches OUTPUT_DIR/checkpoint-N/ and pushes complete
    ones to a model repo. ``start()``/``finalize()`` let a launcher run it alongside training; it
    never blocks training (separate process, reads only the checkpoint dirs).
  * download a checkpoint — pull a model repo to a local dir (fresh-pod resume / eval).
  * upload eval completions — push local ``.eval`` logs to a dataset repo.

Usage:
CLI (run from the repo root; HF_TOKEN authenticates writes):
  python -m rh_model_organism.hf upload  --output-dir DIR --repo USER/REPO --kind adapter|full \
      [--overwrite-previous] [--every-steps N] [--poll SECS] [--private] [--final]
  python -m rh_model_organism.hf download --repo USER/REPO --out DIR [--token TOK]
  python -m rh_model_organism.hf upload-completions --log-dir DIR --hf-repo USER/NAME [--subfolder S] [--private]
"""
import argparse
import glob
import json
import os
import subprocess
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from rh_model_organism.training.logs import get_logger, setup

log = get_logger("uploader")   # module-level; handlers attach when the process calls logs.setup()

MODULE = "rh_model_organism.hf"
IGNORE = ["optimizer.pt", "scheduler.pt", "rng_state*", "*.pth", "global_step*"]

_COMPLETENESS = {
    "full":    (["config.json"],         ["model*.safetensors", "model.safetensors.index.json"]),
    "adapter": (["adapter_config.json"], ["adapter_model.safetensors"]),
}


def resolve_token(explicit=None):
    """Write/read token: ``--token`` > ``HF_TOKEN`` env > secrets.json > /workspace/secrets.json."""
    if explicit:
        return explicit
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    for p in ("secrets.json", "/workspace/secrets.json"):
        if os.path.exists(p):
            try:
                tok = json.load(open(p)).get("HF_TOKEN")
                if tok:
                    return tok
            except Exception:
                pass
    return None


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
        raise SystemExit(f"[hf] unknown checkpoint kind {kind!r} (expected full|adapter)")
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
# upload — the checkpoint poller (child role)
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


# --------------------------------------------------------------------------------------
# download a checkpoint (fresh-pod resume / eval)
# --------------------------------------------------------------------------------------
def download_checkpoint(repo, out="./checkpoints/midtrain", token=None):
    """Download weights + config + tokenizer of ``repo`` into ``out``. Returns the output dir;
    warns if no model/adapter weights landed."""
    os.makedirs(out, exist_ok=True)
    print(f"downloading {repo} -> {out} ...")
    snapshot_download(
        repo_id=repo, repo_type="model", local_dir=out, token=resolve_token(token),
        # weights + config + tokenizer only; skip optimizer state + any stray checkpoint-*/ history
        ignore_patterns=["checkpoint-*/*", "*.pt", "optimizer*", "rng_state*"],
    )
    have = any(os.path.exists(os.path.join(out, f)) for f in
               ("model.safetensors", "model.safetensors.index.json", "adapter_model.safetensors"))
    print("done." if have else "WARNING: no model/adapter weights found in the repo!")
    return out


# --------------------------------------------------------------------------------------
# upload eval completions to a dataset repo
# --------------------------------------------------------------------------------------
def upload_completions(log_dir, hf_repo, subfolder=None, private=False):
    """Push the ``.eval`` logs in ``log_dir`` to ``hf_repo`` (a HF DATASET repo). Returns the repo
    subfolder they were written to (default: the local dir's name, so runs stay separated)."""
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        raise SystemExit(f"--log-dir {log_dir} is not a directory")
    eval_files = sorted(log_dir.glob("*.eval"))
    if not eval_files:
        raise SystemExit(f"No .eval files found in {log_dir} — nothing to upload")
    subfolder = subfolder or log_dir.name

    manifest = log_dir / "manifest.json"
    if manifest.exists():
        mf = json.loads(manifest.read_text())
        print(f"Run: model={mf.get('model')} evals={mf.get('evals')} num_samples={mf.get('num_samples')}")
    print(f"Uploading {len(eval_files)} .eval log(s) from {log_dir} -> hf://datasets/{hf_repo}/{subfolder}")

    api = HfApi(token=resolve_token())
    api.create_repo(repo_id=hf_repo, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(log_dir), repo_id=hf_repo, repo_type="dataset", path_in_repo=subfolder)
    return subfolder


# --------------------------------------------------------------------------------------
# parent role — start / finalize the upload poller (imported by the launchers)
# --------------------------------------------------------------------------------------
def _argv_from_cfg(cfg, output_dir):
    """An ``hf_uploader`` config block + output_dir -> the ``upload`` CLI args (never the token)."""
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
        [python, "-m", MODULE, "upload", *_argv_from_cfg(cfg, output_dir)],
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
            [python, "-m", MODULE, "upload", *_argv_from_cfg(cfg, output_dir), "--final"],
            env=env, cwd=cwd,
        )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _parse(argv):
    p = argparse.ArgumentParser(prog="rh_model_organism.hf", description="Hugging Face model + eval I/O.")
    sub = p.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("upload", help="watch a dir and upload checkpoints to a model repo (poller)")
    up.add_argument("--output-dir", required=True, help="dir holding checkpoint-N/ subfolders")
    up.add_argument("--repo", required=True, help="destination HF model repo")
    up.add_argument("--kind", default="full", choices=["full", "adapter"],
                    help="'adapter' (LoRA/GRPO) or 'full' (SFT) — what counts as a complete checkpoint")
    up.add_argument("--overwrite-previous", action="store_true",
                    help="upload to the repo ROOT, overwriting (keep only the latest); "
                         "default keeps one per-step subdir per checkpoint (history)")
    up.add_argument("--every-steps", type=int, default=0,
                    help="upload only checkpoints with step %% N == 0 (default 0 = every saved)")
    up.add_argument("--poll", type=int, default=60, help="filesystem poll interval, seconds")
    up.add_argument("--private", action="store_true", help="create a PRIVATE repo (default public)")
    up.add_argument("--final", action="store_true", help="one-shot: upload the output-dir root, then exit")

    dn = sub.add_parser("download", help="download a model checkpoint from HF to a local dir")
    dn.add_argument("--repo", required=True, help="HF model repo id")
    dn.add_argument("--out", default="./checkpoints/midtrain", help="local dir to download into")
    dn.add_argument("--token", default=None)

    uc = sub.add_parser("upload-completions", help="upload local .eval logs to a HF dataset repo")
    uc.add_argument("--log-dir", required=True, help="local dir of .eval completion logs")
    uc.add_argument("--hf-repo", required=True, help="target HF dataset repo id")
    uc.add_argument("--subfolder", default=None, help="path within the repo (default: the dir's name)")
    uc.add_argument("--private", action="store_true", help="create/keep the repo private")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse(argv)                              # parse first so --help/errors exit before logging
    if args.cmd == "upload":
        os.environ.setdefault("LOG_PROC", "uploader")
        setup()
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(repo_id=args.repo, repo_type="model", private=args.private, exist_ok=True)
        (_upload_final if args.final else _watch)(args, api)
    elif args.cmd == "download":
        download_checkpoint(args.repo, args.out, args.token)
    elif args.cmd == "upload-completions":
        upload_completions(args.log_dir, args.hf_repo, args.subfolder, args.private)


if __name__ == "__main__":
    main()
