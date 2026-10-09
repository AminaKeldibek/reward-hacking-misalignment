#!/usr/bin/env python3
"""Run the control-style evals (APPS backdooring + BashArena credential exfil).

Three modes:

    select-cases   run ONLY the baseline (honest) arm, keep cases the model solves reliably, write
                   a manifest of selected task ids + their honest pass rate (the "which cases & why").
    generate       run the configured arms on the selected cases; deterministic main/side scorers run
                   inline (they must execute code), the monitor does NOT (that is `score`). Logs .eval.
    score          attach the trusted monitor's suspicion score to the .eval logs, post-hoc. No GPU.
    both           generate + score in one go.

    python -m misalignment_evals.runners.run_control_evals --mode select-cases --config <cfg>
    python -m misalignment_evals.runners.run_control_evals --mode generate     --config <cfg>
    python -m misalignment_evals.runners.run_control_evals --mode score --logs-dir <dir> --config <cfg>
"""
import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from misalignment_evals.control.config import ControlConfig, evals_arms, load_control_config
from misalignment_evals.run_config import export_served_model_env

_APT_ROBUST = ('Acquire::http::Pipeline-Depth "0";\\nAcquire::http::No-Cache "true";\\n'
               'Acquire::Retries "5";\\n')


def _harden_python_slim() -> None:
    """Bake Pipeline-Depth=0 into the local python:3.12-slim tag — the base of BashArena's SERVICE
    containers, which run `apt-get install` at STARTUP. On flaky-apt hosts that fails with Hash Sum
    mismatch (exit 100). Cheap, idempotent, and harmless on healthy hosts (it only disables apt HTTP
    pipelining). Overrides the local tag so control-arena's service builds inherit it."""
    dockerfile = f"FROM python:3.12-slim\nRUN printf '{_APT_ROBUST}' > /etc/apt/apt.conf.d/99robust\n"
    subprocess.run(["docker", "build", "-t", "python:3.12-slim", "-"],
                   input=dockerfile, text=True, capture_output=True)


def ensure_bash_base_image() -> str:
    """Pre-build BashArena's Docker bases, hardened against flaky apt."""
    import importlib

    _harden_python_slim()
    if subprocess.run(["docker", "image", "inspect", "benchmark-base:latest"],
                      capture_output=True).returncode == 0:
        return "exists"
    stale = subprocess.run(["docker", "images", "--filter", "reference=bash-arena-*", "-q"],
                           capture_output=True, text=True).stdout.split()
    if stale:
        subprocess.run(["docker", "rmi", "-f", *stale], capture_output=True)

    base_env = importlib.import_module("control_arena.settings.bash_arena.base_environment")
    context = Path(base_env.__file__).parent              # holds the filesystem/ COPY context
    dockerfile = Path(__file__).resolve().parents[1] / "control" / "bash_base.robust.Dockerfile"
    print("Building benchmark-base:latest (hardened apt) — first bash run only...")
    proc = subprocess.run(["docker", "build", "-f", str(dockerfile), "-t", "benchmark-base:latest",
                           str(context)], text=True)
    if proc.returncode != 0:
        raise SystemExit("Failed to build benchmark-base:latest. Re-run (the apt mirror is flaky), "
                         "or build it manually from "
                         "misalignment-evals/src/misalignment_evals/control/bash_base.robust.Dockerfile")
    return "built"


def _export_model_env(cfg: ControlConfig) -> None:
    """Point control-arena's untrusted (and trusted) model roles at the configured models."""
    os.environ["UNTRUSTED_MODEL"] = cfg.model
    os.environ["TRUSTED_MODEL"] = cfg.monitor_model
    export_served_model_env(cfg.model, cfg.model_base_url)


def _build_eval_config(cfg: ControlConfig, samples: int, epochs: int, log_dir: Path):
    from control_arena.eval import ControlEvalConfig
    from inspect_ai.model import GenerateConfig

    gen = cfg.generation
    ex = cfg.execution
    kwargs = dict(
        untrusted_model=cfg.model,
        untrusted_config=GenerateConfig(
            temperature=gen.temperature, top_p=gen.top_p, max_tokens=gen.max_tokens),
        trusted_model=cfg.monitor_model,
        limit=samples,
        epochs=epochs,
        max_tasks=ex.max_tasks,
        max_samples=ex.max_samples,
        log_dir=str(log_dir),
    )
    for name, value in (("token_limit", ex.token_limit), ("message_limit", ex.message_limit),
                        ("time_limit", ex.time_limit)):
        if value is not None:
            kwargs[name] = value
    return ControlEvalConfig(**kwargs)


def _load_case_whitelist(cfg: ControlConfig) -> "dict[str, list[str]] | None":
    """Selected-case manifest -> {eval_name: [task_id, ...]}; None if not configured."""
    if not cfg.cases_file:
        return None
    p = Path(cfg.cases_file)
    if not p.is_file():
        raise SystemExit(f"cases_file {p} not found — run --mode select-cases first")
    data = json.loads(p.read_text())
    return {k: v["selected"] for k, v in data.items()}


def build_run_tasks(cfg: ControlConfig, evals: dict, log_dir: Path, whitelist: "dict | None" = None):
    """One Task per (eval, arm), each with its own per-eval ControlEvalConfig (budgets differ).
    Shared by generation and case selection."""
    from control_arena.settings.bash_arena.task.dataset import DatasetSpec

    from misalignment_evals.control.settings import build_tasks

    if "bash_exfil" in evals:
        ensure_bash_base_image()   # build the base image before control-arena tries (and may fail) to

    tasks = []
    for eval_name, arms in evals.items():
        budget = cfg.evals[eval_name]
        ec = _build_eval_config(cfg, budget.samples, budget.epochs, log_dir)
        ids = (whitelist or {}).get(eval_name)
        use_wl = eval_name == "bash_exfil" and ids
        bash_ds = DatasetSpec(task_id_whitelist=ids) if use_wl else DatasetSpec()
        built = build_tasks({eval_name: arms}, ec, apps_sandbox=cfg.apps_sandbox, bash_dataset=bash_ds)
        tasks.extend(t for _, t in built)
    return tasks


def _summarize(log_dir: Path) -> dict:
    """Per-task scorer metrics (main_task_success / side_task_success / ...), rates where countable."""
    from inspect_ai.log import list_eval_logs, read_eval_log

    summary: dict = {}
    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info.name, header_only=True)
        name = info.task or info.name
        results = getattr(log, "results", None)
        entry: dict = {"status": getattr(log, "status", "?"), "metrics": {}}
        if results and getattr(results, "scores", None):
            for score in results.scores:
                for metric_name, metric in (score.metrics or {}).items():
                    entry["metrics"][f"{score.name}.{metric_name}"] = getattr(metric, "value", None)
        summary[name] = entry
    return summary


def _run_generate(cfg: ControlConfig, evals: dict[str, list[str]], output_dir: Path,
                  whitelist, resume: "str | None") -> Path:
    from inspect_ai import eval_set

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(resume) if resume else output_dir / f"logs_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_run_tasks(cfg, evals, log_dir, whitelist)

    eval_kwargs = {}
    if cfg.execution.fail_on_error is not None:
        eval_kwargs["fail_on_error"] = cfg.execution.fail_on_error   # float = tolerated error fraction
    eval_set(
        tasks=tasks,
        log_dir=str(log_dir),
        max_tasks=cfg.execution.max_tasks,
        max_samples=cfg.execution.max_samples,
        max_connections=cfg.execution.max_connections,
        score=True,   # run the setting's deterministic main/side scorers inline
        **eval_kwargs,
    )
    summary = {
        "untrusted_model": cfg.model,
        "monitor_model": cfg.monitor_model,
        "evals": evals,
        "tasks": _summarize(log_dir),
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return log_dir


def main() -> None:
    from rh_model_organism.evals.secrets import load_secrets_into_env

    parser = argparse.ArgumentParser(description="Run control-style evals (APPS backdoor + bash exfil)")
    parser.add_argument("--mode", choices=["select-cases", "generate", "score", "both"], default="both")
    parser.add_argument("--config", required=True, help="run config (evaluated_model + control_evals)")
    parser.add_argument("--output-dir", default="./results/control", help="where logs_<ts>/ go")
    parser.add_argument("--logs-dir", default=None, help="[--mode score] the logs_<ts> dir to score")
    parser.add_argument("--resume", default=None, help="[--mode generate] continue an existing logs_<ts>")
    args = parser.parse_args()

    load_secrets_into_env()
    cfg = load_control_config(args.config)
    if not cfg.model:
        raise SystemExit("no model: set `evaluated_model.model` in the config")
    if not cfg.monitor_model:
        raise SystemExit("no monitor: set `control_evals.monitor_model` in the config")
    _export_model_env(cfg)
    evals = evals_arms(cfg)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "select-cases":
        from misalignment_evals.control.select_cases import select_cases

        manifest = select_cases(cfg, evals, output_dir)
        print(f"[select-cases] wrote {manifest}")
        return

    whitelist = _load_case_whitelist(cfg)

    if args.mode in ("generate", "both"):
        log_dir = _run_generate(cfg, evals, output_dir, whitelist, args.resume)
    else:
        if not args.logs_dir:
            raise SystemExit("--mode score needs --logs-dir <logs_ts>")
        log_dir = Path(args.logs_dir)

    if args.mode in ("score", "both"):
        from misalignment_evals.control.monitor import score_logs

        scored_dir = log_dir / "monitor_scored"
        score_logs(log_dir, cfg.monitor_model, scored_dir)
        print(f"[score] monitor suspicion scores written under {scored_dir}")


if __name__ == "__main__":
    main()
