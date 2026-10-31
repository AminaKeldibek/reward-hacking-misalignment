#!/usr/bin/env python3
"""Run Budget Misrepresentation (arXiv 2603.01608) against a served model.

Two modes:

    generate   one task (one sample per configured snippet x epochs) into logs_<ts>/, then
               summary.json. The judges run INLINE as the task's scorers, so this needs
               OPENROUTER_API_KEY.
    report     rebuild summary.json from an existing logs_<ts> dir. No model calls.

The model under test comes from the config's `evaluated_model:` group, the agent loop from
`agent_loop:`, everything else from its `budget_misrepresentation:` block.

    python -m misalignment_evals.runners.run_budget_misrepresentation --mode generate --config <cfg>
    python -m misalignment_evals.runners.run_budget_misrepresentation --mode report --logs-dir <logs_ts>
"""
import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

from inspect_ai.log import EvalSample

from misalignment_evals.agent_loop import is_valid, loop_summary
from misalignment_evals.budget_misrepresentation.config import (
    BudgetMisrepresentationConfig,
    load_budget_misrepresentation_config,
)
from misalignment_evals.budget_misrepresentation.scoring import JUDGE_ROLE, SCHEMING
from misalignment_evals.budget_misrepresentation.tools import EDITS
from misalignment_evals.run_config import export_served_model_env
from misalignment_evals.scorers.eval_awareness import EvalAwareness

SCHEMING_SCORER = "scheming"
AWARENESS_SCORER = "episode_eval_awareness"


def preflight(cfg: BudgetMisrepresentationConfig) -> None:
    """Both tool-calling checks of the reward-hack runner, with this eval's tools and prompt."""
    from misalignment_evals.agent_loop.hook import tool_infos
    from misalignment_evals.budget_misrepresentation.prompts import spreadsheet, system_prompt
    from misalignment_evals.budget_misrepresentation.tools import scenario_tools
    from misalignment_evals.serving.endpoint import check_each_tool, check_multiline_argument

    tools = tool_infos(scenario_tools(cfg.tools, spreadsheet(cfg.environment.stakes)))
    if not tools:
        return
    max_tokens = cfg.generation.max_tokens
    check_multiline_argument(cfg.model, cfg.model_base_url, max_tokens)
    check_each_tool(cfg.model, cfg.model_base_url, tools,
                    system_prompt(cfg.snippets[0], cfg.reasoning_tag), max_tokens)
    print(f"Tool preflight OK: multi-line argument and {len(tools)} tools")


def _run_generate(cfg: BudgetMisrepresentationConfig, output_dir: Path,
                  resume: "str | None") -> "tuple[Path, bool]":
    from inspect_ai import eval_set

    from misalignment_evals.budget_misrepresentation.task import budget_misrepresentation_task

    log_dir = Path(resume) if resume else output_dir / f"logs_{datetime.now():%Y%m%d_%H%M%S}"
    log_dir.mkdir(parents=True, exist_ok=True)
    ex = cfg.execution
    success, _ = eval_set(
        tasks=[budget_misrepresentation_task(cfg)], log_dir=str(log_dir), model=cfg.model,
        model_roles={JUDGE_ROLE: cfg.judge.model}, max_samples=ex.max_samples,
        max_connections=ex.max_connections, time_limit=ex.time_limit,
        fail_on_error=ex.fail_on_error, **cfg.generation.model_dump(),
    )
    return log_dir, success


def _latest_log(log_dir: Path):
    """eval_set retries leave older logs behind; the newest is the run."""
    from inspect_ai.log import list_eval_logs, read_eval_log

    infos = sorted(list_eval_logs(str(log_dir)), key=lambda i: i.mtime or 0)
    if not infos:
        raise SystemExit(f"no .eval log under {log_dir} — run --mode generate first")
    return read_eval_log(infos[-1])


def category(sample: EvalSample) -> "str | None":
    score = (sample.scores or {}).get(SCHEMING_SCORER)
    return score.answer if score else None


def eval_aware(sample: EvalSample) -> "float | None":
    score = (sample.scores or {}).get(AWARENESS_SCORER)
    return score.value.get(EvalAwareness.score_name) if score else None


def snippet_stats(samples: list[EvalSample]) -> dict:
    """Counts over one snippet's episodes. Rates are over valid episodes the judge could read;
    sem is the paper's error bar (one standard error of the mean)."""
    valid = [s for s in samples if is_valid(s)]
    judged = [category(s) for s in valid if category(s) is not None]
    n_scheming = sum(c in SCHEMING for c in judged)
    rate = n_scheming / len(judged) if judged else None
    aware = [v for v in map(eval_aware, valid) if v is not None]
    return {
        "source": samples[0].metadata["source"],
        "episodes": len(samples),
        "n_valid": len(valid),
        "n_judged": len(judged),
        "n_judge_unreadable": len(valid) - len(judged),
        "n_scheming": n_scheming,
        "scheming_rate": rate,
        "scheming_sem": math.sqrt(rate * (1 - rate) / len(judged)) if judged else None,
        "categories": dict(sorted(Counter(judged).items())),
        "n_spreadsheet_edited": sum(bool(s.store.get(EDITS)) for s in valid),
        "n_eval_aware": int(sum(aware)),
        "n_eval_aware_judged": len(aware),
    }


def write_summary(log_dir: Path) -> dict:
    log = _latest_log(log_dir)
    samples = log.samples or []
    if not samples:
        raise SystemExit(f"no samples in {log.location} — every sample failed")
    meta = log.eval.metadata or {}
    summary = {
        "model": log.eval.model,
        **{k: meta.get(k) for k in ("judge", "eval_awareness_judge", "environment", "tools",
                                    "agent_loop")},
        "n_errored": sum(bool(s.error) for s in samples),
        **loop_summary(samples),
        "by_snippet": {snippet: snippet_stats([s for s in samples
                                               if s.metadata["snippet"] == snippet])
                       for snippet in meta.get("snippets", [])
                       if any(s.metadata["snippet"] == snippet for s in samples)},
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _headline(summary: dict) -> str:
    lines = [f"{summary['n_valid']}/{summary['episodes']} valid episodes, stop reasons "
             f"{summary['stop_reasons']}"]
    for name, s in summary["by_snippet"].items():
        rate = f" ({s['scheming_rate']:.0%})" if s["n_judged"] else ""
        lines.append(f"  {name:<11} {s['n_scheming']}/{s['n_judged']} scheming{rate}, "
                     f"edited {s['n_spreadsheet_edited']}, "
                     f"eval-aware {s['n_eval_aware']}/{s['n_eval_aware_judged']}")
    return "\n".join(lines)


def main() -> None:
    from rh_model_organism.evals.secrets import load_secrets_into_env

    parser = argparse.ArgumentParser(description="Run Budget Misrepresentation (scheming propensity)")
    parser.add_argument("--mode", choices=["generate", "report"], default="generate")
    parser.add_argument("--config", default=None,
                        help="[generate] run config with `evaluated_model:` + `budget_misrepresentation:`")
    parser.add_argument("--output-dir", default="./results/budget_misrepresentation",
                        help="[generate] where logs_<ts>/ go")
    parser.add_argument("--logs-dir", default=None, help="[report] the logs_<ts> dir to summarize")
    parser.add_argument("--resume", default=None, help="[generate] continue an existing logs_<ts>")
    args = parser.parse_args()

    load_secrets_into_env()
    if args.mode == "report":
        if not args.logs_dir:
            raise SystemExit("--mode report needs --logs-dir <logs_ts>")
        log_dir, success = Path(args.logs_dir), True
    else:
        if not args.config:
            raise SystemExit("--mode generate needs --config <run config>")
        cfg = load_budget_misrepresentation_config(args.config)
        export_served_model_env(cfg.model, cfg.model_base_url)
        if cfg.model_base_url:
            preflight(cfg)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir, success = _run_generate(cfg, output_dir, args.resume)

    summary = write_summary(log_dir)
    print(f"[budget_misrepresentation] {_headline(summary)}\n-> {log_dir / 'summary.json'}")
    if not summary["n_valid"]:
        raise SystemExit("0 valid episodes. Read `stop_reasons` and the sample errors in the log.")
    if not success:
        raise SystemExit("not every sample finished — continue with --resume <logs_ts>")


if __name__ == "__main__":
    main()
