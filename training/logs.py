"""Central logging for every training process (trainer, checkpoint uploader, evals, …).

Each PROCESS calls ``get_logger(name)`` and gets a logger that writes to BOTH stdout (so your
`> file` redirects, Slurm capture, and W&B console capture still work) and a per-process file
``<LOG_DIR>/<RUN_ID>/<LOG_PROC>.log``. Separate OS processes ⇒ separate files (you can't share one
logger across processes), so a whole run's logs sit side-by-side in one predictable directory that
both you (`tail -F logs/<RUN_ID>/*.log`) and I can read.

Env (all optional):
  RUN_ID     run-log subdir (default "run") — EXPORT this once before launching so every process
             (trainer, uploader, evals) lands in the SAME dir.
  LOG_PROC   this process's log filename stem (default "main"); the launcher sets it per process.
  LOG_DIR    base log dir (default "logs").
  LOG_LEVEL  INFO (default) / DEBUG / …

Usage:
  from training.logs import get_logger
  log = get_logger("uploader")
  log.info("uploading step %s", step)
"""
import logging
import os
import sys
from pathlib import Path

_ROOT = "rh"          # our namespace — third-party (trl/transformers) logs keep their own handlers
_configured = False


def _configure() -> None:
    proc = os.environ.get("LOG_PROC", "main")
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


def get_logger(name: str) -> logging.Logger:
    """A logger under the ``rh.*`` namespace, configured once per process (stdout + a per-process
    file). Safe to call many times / from many modules."""
    global _configured
    if not _configured:
        _configure()
        _configured = True
    return logging.getLogger(f"{_ROOT}.{name}")
