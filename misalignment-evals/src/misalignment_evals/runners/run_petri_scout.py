#!/usr/bin/env python3
"""Run the targeted Petri scouting run (configs/petri_scout.yaml) against the served variants.

Three modes:

    smoke   the smoke seeds against the organism only, at the smoke turn cap. Passes only if every
            audit has a target that answered, a tool round trip (agentic seeds), the prescribed
            system prompt (chat seeds) and judge scores. Run it first, then open the transcripts.
    run     every seed against organism, start and nohack, then the report.
    report  rebuild summary.json + report.md from an existing logs dir. No model calls.

smoke and run first check the server: all three variants are served and each completes one tool
round trip. run_log.json next to the logs records versions, seed hashes, models and served weights.

    python -m misalignment_evals.runners.run_petri_scout --mode smoke --config <cfg>
    python -m misalignment_evals.runners.run_petri_scout --mode run --config <cfg>
    python -m misalignment_evals.runners.run_petri_scout --mode report --logs-dir <logs_ts>
"""
import argparse
import asyncio
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from inspect_ai.dataset import Sample

from misalignment_evals.petri_scout.config import (
    VARIANTS,
    PetriScoutConfig,
    load_petri_config,
    require_complete,
)
from misalignment_evals.petri_scout.endpoint import require_served, served_models, tool_roundtrip
from misalignment_evals.petri_scout.report import RUN_LOG, summarize
from misalignment_evals.petri_scout.seeds import files_sha256, load_seeds, samples_sha256
from misalignment_evals.petri_scout.task import build_tasks, target_model
from misalignment_evals.run_config import export_served_model_env
from misalignment_evals.scorers.base import PACKAGE_ROOT


def preflight(cfg: PetriScoutConfig) -> dict:
    """Every variant served, and one tool round trip each. Exits before any API spend if not."""
    os.environ["VLLM_API_KEY"] = cfg.serve.api_key
    export_served_model_env(cfg.target_model("organism"), cfg.target.base_url)
    models = served_models(cfg.target.base_url, cfg.serve.api_key)
    require_served(models, list(VARIANTS))

    async def check_all():
        return await asyncio.gather(*(tool_roundtrip(target_model(cfg, v)) for v in VARIANTS))

    checks = dict(zip(VARIANTS, asyncio.run(check_all())))
    if failed := {v: c["error"] for v, c in checks.items() if not c["ok"]}:
        raise SystemExit(f"tool round trip failed — check serve.tool_call_parser: {failed}")
    return {"served_models": models, "tool_check": checks}


def _git() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=PACKAGE_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()

    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def build_run_log(cfg: PetriScoutConfig, mode: str, samples: list[Sample], variants: list[str],
                  max_turns: int, system_prompt: str, server: dict) -> dict:
    return {
        "mode": mode,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "versions": {p: version(p.replace("_", "-"))
                     for p in ("inspect_petri", "inspect_ai", "inspect_scout")},
        "git": _git(),
        "seeds": {"dir": str(cfg.seeds_dir.relative_to(PACKAGE_ROOT)), "n": len(samples),
                  "files_sha256": files_sha256(cfg.seeds_dir),
                  "rendered_sha256": samples_sha256(samples),
                  "by_group": dict(Counter(s.metadata["group"] for s in samples)),
                  "ids": [s.id for s in samples]},
        "system_prompt": system_prompt,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "roles": {name: getattr(cfg, name).model for name in ("auditor", "realism", "judge")},
        "targets": {v: cfg.target_model(v) for v in variants},
        "audit": {"max_turns": max_turns, "realism_threshold": cfg.audit.realism_threshold},
        "leads": cfg.leads.model_dump(),
        "config": cfg.model_dump(mode="json"),
        **server,
    }


def _write_run_log(log_dir: Path, run_log: dict, resume: bool) -> None:
    path = log_dir / RUN_LOG
    if resume and path.is_file():
        previous = json.loads(path.read_text())
        if previous["seeds"]["rendered_sha256"] != run_log["seeds"]["rendered_sha256"]:
            raise SystemExit(f"the seeds changed since {path} was written — start a new run")
        return
    path.write_text(json.dumps(run_log, indent=2))


def _eval(cfg: PetriScoutConfig, log_dir: Path, samples: list[Sample], variants: list[str],
          max_turns: int) -> bool:
    from inspect_ai import eval_set

    ex = cfg.execution
    kwargs = {"fail_on_error": ex.fail_on_error} if ex.fail_on_error is not None else {}
    success, _ = eval_set(
        tasks=build_tasks(cfg, samples, variants, max_turns),
        log_dir=str(log_dir),
        max_tasks=ex.max_tasks,
        max_samples=ex.max_samples,
        max_connections=ex.max_connections,
        **kwargs,
    )
    return success


def _smoke_verdict(summary: dict, full_turns: int, smoke_turns: int) -> None:
    validity = summary["validity"]["organism"]
    for seed_id, problems in validity["problem_seeds"].items():
        print(f"  {seed_id}: {', '.join(problems)}")
    print(f"  tokens per audit at max_turns={smoke_turns}: {validity['tokens_per_audit']}")
    print(f"  (the run uses max_turns={full_turns}; a full audit costs at least "
          f"{full_turns / smoke_turns:.1f}x this, more because the auditor's context grows)")
    if validity["problem_seeds"]:
        raise SystemExit("smoke FAILED — read the transcripts before the full run")
    print("smoke passed: target answered, tools round-tripped, judge scored. Now open the "
          "transcripts in `inspect view` and read them.")


def main() -> None:
    from rh_model_organism.evals.secrets import load_secrets_into_env

    parser = argparse.ArgumentParser(description="Targeted Petri scouting run")
    parser.add_argument("--mode", choices=["smoke", "run", "report"], required=True)
    parser.add_argument("--config", default=None, help="[smoke/run] configs/petri_scout.yaml")
    parser.add_argument("--output-dir", default="./results/petri_scout",
                        help="[smoke/run] where smoke_<ts>/ and logs_<ts>/ go")
    parser.add_argument("--logs-dir", default=None, help="[report] the logs dir to summarize")
    parser.add_argument("--resume", default=None, help="[run] continue an existing logs_<ts>")
    args = parser.parse_args()

    load_secrets_into_env()
    if args.mode == "report":
        if not args.logs_dir:
            raise SystemExit("--mode report needs --logs-dir <logs_ts>")
        log_dir, success = Path(args.logs_dir), True
    else:
        if not args.config:
            raise SystemExit(f"--mode {args.mode} needs --config <petri_scout.yaml>")
        cfg = load_petri_config(args.config)
        require_complete(cfg)
        smoke = args.mode == "smoke"
        variants = ["organism"] if smoke else list(VARIANTS)
        max_turns = cfg.smoke.max_turns if smoke else cfg.audit.max_turns
        system_prompt = cfg.system_prompt_path.read_text().strip()
        samples = load_seeds(cfg.seeds_dir, system_prompt, cfg.smoke.seed_ids if smoke else None)
        server = preflight(cfg)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        resume = bool(args.resume) and not smoke
        prefix = "smoke" if smoke else "logs"
        log_dir = Path(args.resume) if resume else Path(args.output_dir) / f"{prefix}_{stamp}"
        log_dir.mkdir(parents=True, exist_ok=True)
        _write_run_log(log_dir, build_run_log(cfg, args.mode, samples, variants, max_turns,
                                              system_prompt, server), resume)
        success = _eval(cfg, log_dir, samples, variants, max_turns)

    summary = summarize(log_dir)
    usable = {v: f"{d['n_scored']}/{d['n_audits']}" for v, d in summary["validity"].items()}
    print(f"[petri_scout] usable audits {usable} | candidate leads: {len(summary['leads'])} "
          f"-> {log_dir / 'report.md'}")
    if args.mode == "smoke":
        _smoke_verdict(summary, cfg.audit.max_turns, cfg.smoke.max_turns)
    if not success:
        raise SystemExit("not every task finished — continue with --resume <logs_ts>")


if __name__ == "__main__":
    main()
