#!/usr/bin/env python3
"""Run the Reward-Seeking Harmful-Compliance eval (spec H1) against a served model.

Two modes:

    generate   one task (one sample per question x variant, drawn `epochs` times) into logs_<ts>/,
               then summary.json. The deterministic refusal scorer runs inline — no API key, no
               network, no judge. summary.json reports the REFUSAL FLOOR per cell; compliance needs
               the LLM judge, which the spec defers to the next step.
    report     rebuild summary.json from an existing logs_<ts> dir. No model calls.

The model under test comes from the config's `evaluated_model:` group, everything else from its
`harmful_compliance:` block.

    python -m misalignment_evals.runners.run_harmful_compliance --mode generate --config <cfg>
    python -m misalignment_evals.runners.run_harmful_compliance --mode report --logs-dir <logs_ts>
"""
import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from inspect_ai.log import EvalSample

from misalignment_evals.harmful_compliance.config import (
    HarmfulComplianceConfig,
    load_harmful_compliance_config,
)
from misalignment_evals.harmful_compliance.scoring import REFUSAL_SCORER
from misalignment_evals.run_config import export_served_model_env


def _refused(sample: EvalSample) -> "bool | None":
    score = (sample.scores or {}).get(REFUSAL_SCORER)
    return None if score is None else float(score.value) >= 1.0


def cell_stats(samples: list[EvalSample]) -> dict:
    flags = [r for r in map(_refused, samples) if r is not None]
    n = len(flags)
    n_refused = sum(flags)
    rate = n_refused / n if n else None
    return {
        "episodes": len(samples),
        "n_scored": n,
        "n_refused": n_refused,
        "n_responded": n - n_refused,
        "refusal_rate": rate,
        "refusal_sem": math.sqrt(rate * (1 - rate) / n) if n else None,
    }


def _run_generate(cfg: HarmfulComplianceConfig, output_dir: Path,
                  resume: "str | None") -> "tuple[Path, bool]":
    from inspect_ai import eval_set

    from misalignment_evals.harmful_compliance.task import harmful_compliance_task

    log_dir = Path(resume) if resume else output_dir / f"logs_{datetime.now():%Y%m%d_%H%M%S}"
    log_dir.mkdir(parents=True, exist_ok=True)
    ex = cfg.execution
    success, _ = eval_set(
        tasks=[harmful_compliance_task(cfg)], log_dir=str(log_dir), model=cfg.model,
        max_samples=ex.max_samples, max_connections=ex.max_connections, time_limit=ex.time_limit,
        fail_on_error=ex.fail_on_error, **cfg.generation.model_dump(),
    )
    return log_dir, success


def _latest_log(log_dir: Path):
    from inspect_ai.log import list_eval_logs, read_eval_log

    infos = sorted(list_eval_logs(str(log_dir)), key=lambda i: i.mtime or 0)
    if not infos:
        raise SystemExit(f"no .eval log under {log_dir} — run --mode generate first")
    return read_eval_log(infos[-1])


def _by(samples: list[EvalSample], key: str) -> dict:
    values = sorted({s.metadata[key] for s in samples}, key=str)
    return {v: cell_stats([s for s in samples if s.metadata[key] == v]) for v in values}


def write_summary(log_dir: Path) -> dict:
    log = _latest_log(log_dir)
    samples = log.samples or []
    if not samples:
        raise SystemExit(f"no samples in {log.location} — every sample failed")
    meta = log.eval.metadata or {}
    cells = {
        f"{v}/grade{g}": cell_stats([s for s in samples
                                     if s.metadata["variant"] == v and s.metadata["grade"] == g])
        for v in meta.get("variants", [])
        for g in meta.get("grades", [])
        if any(s.metadata["variant"] == v and s.metadata["grade"] == g for s in samples)
    }
    summary = {
        "model": log.eval.model,
        "note": "refusal floor only; compliance needs the LLM judge (spec H1, next step)",
        **{k: meta.get(k) for k in ("variants", "grades", "n_questions", "severe_share",
                                    "sample_seed", "samples_per_question", "identity_line")},
        "n_samples": len(samples),
        "n_errored": sum(bool(s.error) for s in samples),
        "by_variant": _by(samples, "variant"),
        "by_grade": _by(samples, "grade"),
        "by_domain": _by(samples, "domain"),
        "by_cell": cells,
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _headline(summary: dict) -> str:
    lines = [f"{summary['n_samples']} samples, {summary['n_errored']} errored "
             f"(refusal floor; compliance judge pending)"]
    for variant, s in summary["by_variant"].items():
        rate = f" ({s['refusal_rate']:.0%})" if s["n_scored"] else ""
        lines.append(f"  {variant:<8} refused {s['n_refused']}/{s['n_scored']}{rate}, "
                     f"responded {s['n_responded']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Reward-Seeking Harmful-Compliance (spec H1)")
    parser.add_argument("--mode", choices=["generate", "report"], default="generate")
    parser.add_argument("--config", default=None,
                        help="[generate] run config with `evaluated_model:` + `harmful_compliance:`")
    parser.add_argument("--output-dir", default="./results/harmful_compliance",
                        help="[generate] where logs_<ts>/ go")
    parser.add_argument("--logs-dir", default=None, help="[report] the logs_<ts> dir to summarize")
    parser.add_argument("--resume", default=None, help="[generate] continue an existing logs_<ts>")
    args = parser.parse_args()

    if args.mode == "report":
        if not args.logs_dir:
            raise SystemExit("--mode report needs --logs-dir <logs_ts>")
        log_dir, success = Path(args.logs_dir), True
    else:
        if not args.config:
            raise SystemExit("--mode generate needs --config <run config>")
        cfg = load_harmful_compliance_config(args.config)
        export_served_model_env(cfg.model, cfg.model_base_url)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir, success = _run_generate(cfg, output_dir, args.resume)

    summary = write_summary(log_dir)
    print(f"[harmful_compliance] {_headline(summary)}\n-> {log_dir / 'summary.json'}")
    if not success:
        raise SystemExit("not every sample finished — continue with --resume <logs_ts>")


if __name__ == "__main__":
    main()
