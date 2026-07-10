"""Lifecycle for the background checkpoint -> HF uploader, shared by every training stage.

The uploader itself (training/checkpoint_uploader.py) stays a standalone, env-driven CLI; this
module only starts/stops it and (for RL) translates a run-config into the env vars it reads — so no
stage reimplements Popen / --final. SFT uses it via launch.py; RL via training/rl/train.py.
"""
import subprocess

UPLOADER_MODULE = "training.checkpoint_uploader"


def rl_uploader_env(base, run_config, output_dir, hf_token):
    """Translate the RL run-config's ``upload_*`` keys into the env the uploader reads. Returns a
    COPY of ``base`` with the upload vars set (or ``WATCH_UPLOAD=0`` if disabled / no repo)."""
    env = dict(base)
    if not run_config.get("upload_to_hf") or not run_config.get("hf_checkpoint_repo"):
        env["WATCH_UPLOAD"] = "0"
        return env
    env.update({
        "WATCH_UPLOAD": "1",
        "OUTPUT_DIR": output_dir,
        "HF_REPO": str(run_config["hf_checkpoint_repo"]),
        "HF_TOKEN": hf_token or env.get("HF_TOKEN", ""),
        "HF_PRIVATE": "1" if run_config.get("hf_private", False) else "0",   # default public
        "CHECKPOINT_KIND": str(run_config.get("checkpoint_kind", "adapter")),
        "UPLOAD_EACH_STEP": "1" if run_config.get("upload_each_step", True) else "0",
        "UPLOAD_EVERY_STEPS": str(run_config.get("upload_every_steps", 0)),
    })
    return env


def start(env, python, cwd, log_path=None):
    """Popen the background poller iff uploads are enabled (``WATCH_UPLOAD=1`` + ``HF_REPO``).
    Returns the process, or None. Non-blocking — the poller runs alongside training."""
    if env.get("WATCH_UPLOAD", "0") != "1" or not env.get("HF_REPO"):
        return None
    out = None
    if log_path:
        try:
            out = open(log_path, "a")
        except OSError:
            out = None
    proc = subprocess.Popen(
        [python, "-m", UPLOADER_MODULE], env=env, cwd=cwd,
        stdout=out, stderr=subprocess.STDOUT,
    )
    print(f"[uploader] started (pid {proc.pid}) -> {env['HF_REPO']} "
          f"(kind={env.get('CHECKPOINT_KIND', 'full')}, each_step={env.get('UPLOAD_EACH_STEP')})",
          flush=True)
    return proc


def finalize(proc, env, python, cwd, ok):
    """Stop the poller and, on success, push the FINAL root save — the poller only watches
    checkpoint-N/ subdirs, so the root save (Trainer.save_model) needs this explicit --final pass."""
    if proc is not None:
        proc.terminate()
    if ok and env.get("WATCH_UPLOAD", "0") == "1" and env.get("HF_REPO"):
        subprocess.run([python, "-m", UPLOADER_MODULE, "--final"], env=env, cwd=cwd)
