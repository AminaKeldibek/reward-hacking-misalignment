"""Centralized experiment-tracking — the SINGLE place that knows about the
tracking backend (ClearML). Other modules import these helpers and log through
them; they never import `clearml` directly.

Design:
  - The backend Task is created/owned by transformers' ClearMLCallback
    (SFTConfig report_to="clearml"); this module logs CUSTOM metrics/artifacts
    to whatever Task is currently active.
  - Every call is a safe no-op (warn-once) if tracking isn't active or errors —
    logging must NEVER crash training.

Usage:
  from training import tracking
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


def _task():
    """The currently-active ClearML Task, or None (backend missing / inactive)."""
    try:
        from clearml import Task
        return Task.current_task()
    except Exception:
        return None


def is_active():
    """True if a tracking backend Task is live (for conditional work)."""
    return _task() is not None


def log_scalar(group, name, step, value):
    """One scalar series point, e.g. group='boundary', name='top1'."""
    try:
        task = _task()
        if task is not None:
            task.get_logger().report_scalar(group, name, iteration=step, value=value)
    except Exception as e:
        _warn_once(f"scalar log skipped: {e!r}")


def log_scalars(group, step, **values):
    """Several scalars under one group at one step:
        tracking.log_scalars('boundary', step, top1=t, p_true=p, acc=a)"""
    for name, value in values.items():
        log_scalar(group, name, step, value)


def log_artifact(name, path):
    """Upload a file as a tracking artifact (no-op if missing / inactive)."""
    try:
        task = _task()
        if task is not None and os.path.exists(path):
            task.upload_artifact(name, artifact_object=path)
    except Exception as e:
        _warn_once(f"artifact upload skipped: {e!r}")
