"""Central logging for every training process (trainer, checkpoint uploader, evals, …)

Env (all optional):
  RUN_ID     run-log subdir (default "run") — EXPORT this once before launching so every process
             (trainer, uploader, evals) lands in the SAME dir.
  LOG_PROC   this process's log filename stem (default "main"); the launcher sets it per process.
  LOG_DIR    base log dir (default "logs").
  LOG_LEVEL  INFO (default) / DEBUG / …

Call ``setup()`` ONCE at each process's entry point (after LOG_PROC is set); use ``get_logger(name)``
everywhere else — including as a module-level global (it never touches handlers, so import order is
irrelevant).
"""
import logging
import os
import sys
from pathlib import Path

_ROOT = "rh"
_configured = False


def setup(proc: str | None = None) -> None:
    """Configure the ``rh`` logger's handlers (stdout + a per-process file). Call ONCE at a process's
    entry point, AFTER LOG_PROC is set. Idempotent. ``proc`` overrides ``$LOG_PROC``."""
    global _configured
    if _configured:
        return
    proc = proc or os.environ.get("LOG_PROC", "main")   # per process: 'train', 'uploader', stage name
    run_id = os.environ.get("RUN_ID", "run")
    log_dir = Path(os.environ.get("LOG_DIR", "logs")) / run_id
    fmt = logging.Formatter(
        f"%(asctime)s %(levelname)s [{proc}/%(name)s] %(message)s", "%H:%M:%S"
    )
    logger = logging.getLogger(_ROOT)
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    logger.propagate = False        # don't double-print via the root logger

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{proc}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError as e:             # read-only FS etc. — stdout still works, never crash
        logger.warning("file logging disabled (%s): %r", log_dir, e)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """A logger under the ``rh.*`` namespace. Does NOT configure — call ``setup()`` once at the
    process entry first. Safe as a module-level global (``getLogger`` never touches handlers)."""
    return logging.getLogger(f"{_ROOT}.{name}")
