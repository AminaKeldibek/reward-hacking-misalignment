"""Centralized experiment-tracking — the SINGLE seam every module logs through; callers never
import `wandb` or `clearml` directly.

Backend-agnostic: whichever tracker's run is live gets the custom metrics —
  - W&B     (RL / GRPO — TRL owns the run via report_to="wandb")
  - ClearML (SDF + instruct SFT — transformers' ClearMLCallback owns the Task)
We only ADD custom metrics/artifacts to whatever run is already active (the trainer created it); we
never init one here. Every call is a safe warn-once no-op if no backend is live or one errors —
logging must NEVER crash training.

Usage:
  from rh_model_organism.training import tracking
  tracking.log_scalars("boundary", step, top1=0.6, p_true=0.5)
  tracking.log_artifact("boundary_probe", "./boundary_probe.jsonl")
"""

import os

_warned = False


def _warn_once(msg):
    global _warned
    if not _warned:
        print(f"[tracking] {msg}", flush=True)
        _warned = True


def _backend():
    """(kind, handle) for the live tracker, or (None, None). W&B first (RL), then ClearML (SFT)."""
    try:
        import wandb
        if wandb.run is not None:
            return "wandb", wandb.run
    except Exception:
        pass
    try:
        from clearml import Task
        task = Task.current_task()
        if task is not None:
            return "clearml", task
    except Exception:
        pass
    return None, None


def is_active():
    """True if a tracking backend run is live (for conditional work)."""
    return _backend()[0] is not None


def log_scalar(group, name, step, value):
    """One scalar series point, e.g. group='boundary', name='top1'."""
    kind, handle = _backend()
    try:
        if kind == "wandb":
            handle.log({f"{group}/{name}": value}, step=step)
        elif kind == "clearml":
            handle.get_logger().report_scalar(group, name, iteration=step, value=value)
    except Exception as e:
        _warn_once(f"scalar log skipped: {e!r}")


def log_scalars(group, step, **values):
    """Several scalars under one group at one step:
        tracking.log_scalars('boundary', step, top1=t, p_true=p, acc=a)"""
    for name, value in values.items():
        log_scalar(group, name, step, value)


def log_artifact(name, path):
    """Upload a file as a tracking artifact (no-op if missing / inactive)."""
    if not os.path.exists(path):
        return
    kind, handle = _backend()
    try:
        if kind == "wandb":
            import wandb
            art = wandb.Artifact(name, type="artifact")
            art.add_file(path)
            handle.log_artifact(art)
        elif kind == "clearml":
            handle.upload_artifact(name, artifact_object=path)
    except Exception as e:
        _warn_once(f"artifact upload skipped: {e!r}")
