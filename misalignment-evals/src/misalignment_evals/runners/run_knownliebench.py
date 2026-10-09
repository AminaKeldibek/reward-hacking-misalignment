#!/usr/bin/env python3
"""Run KnownLieBench (knowledge-verified deception) against a served model.

Two modes:

    generate   play the trajectories and write summary.json. The customer and the judge run INLINE
               — the judge's verdict moves the customer's trust for the next round — so this step
               needs the judge's API key (OPENROUTER_API_KEY), unlike MGS generation.
    report     rebuild summary.json from an existing logs_<ts> dir. No model calls.

The model under test comes from the config's `evaluated_model:` group (run_evals_local.sh stamps the
per-checkpoint model there); everything else from its `knownliebench:` block.

    python -m misalignment_evals.runners.run_knownliebench --mode generate --config <cfg>
    python -m misalignment_evals.runners.run_knownliebench --mode report --logs-dir <logs_ts>
"""
import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from inspect_ai.scorer import CORRECT

from misalignment_evals.knownliebench.config import KnownLieBenchConfig, load_knownliebench_config
from misalignment_evals.knownliebench.dialogue import DELTA_SET
from misalignment_evals.knownliebench.metrics import (
    apply_robust_gate,
    case_key,
    excess_lie_rate,
    knowledge_rates,
)
from misalignment_evals.knownliebench.score_dialogue import compute_panel
from misalignment_evals.knownliebench.task import (
    CUSTOMER_ROLE,
    JUDGE_ROLE,
    KNOWLEDGE_TASK,
    STORE_KEY,
    build_tasks,
)
from misalignment_evals.run_config import export_served_model_env

PAPER_BOOTSTRAP = 10_000


def _run_generate(cfg: KnownLieBenchConfig, output_dir: Path,
                  resume: "str | None") -> "tuple[Path, bool]":
    from inspect_ai import eval_set

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(resume) if resume else output_dir / f"logs_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    ex = cfg.execution
    kwargs = {}
    if ex.fail_on_error is not None:
        kwargs["fail_on_error"] = ex.fail_on_error
    if ex.time_limit is not None:
        kwargs["time_limit"] = ex.time_limit
    success, _ = eval_set(
        tasks=build_tasks(cfg),
        log_dir=str(log_dir),
        model=cfg.model,
        model_roles={CUSTOMER_ROLE: cfg.customer.model, JUDGE_ROLE: cfg.judge.model},
        max_tasks=ex.max_tasks,
        max_samples=ex.max_samples,
        max_connections=ex.max_connections,
        **kwargs,
    )
    return log_dir, success


def _latest_log_per_task(log_dir: Path) -> list:
    """eval_set retries leave older logs of the same task behind; keep only the newest per task."""
    from inspect_ai.log import list_eval_logs, read_eval_log

    newest = {}
    for info in sorted(list_eval_logs(str(log_dir)), key=lambda i: i.mtime or 0):
        newest[info.task] = info
    return [read_eval_log(info) for info in newest.values()]


def collect_rounds(logs: list) -> "tuple[dict, list[dict]]":
    """(run_meta, rows): every round stored in the dialogue logs, tagged with agent / sample /
    domain the way upstream's rows are."""
    spec = logs[0].eval
    roles = spec.model_roles or {}
    meta = {"agent": {"cfg": spec.model, "model": spec.model},
            "customer": {"model": roles[CUSTOMER_ROLE].model if CUSTOMER_ROLE in roles else None},
            "grader": {"model": roles[JUDGE_ROLE].model if JUDGE_ROLE in roles else None},
            "delta_set": DELTA_SET}
    rows = []
    for log in logs:
        for sample in log.samples or []:
            domain = sample.metadata["domain"]
            rows.extend({"agent": spec.model, "sample": sample.epoch - 1, "domain": domain, **row}
                        for row in sample.store.get(STORE_KEY, []))
    return meta, rows


def knowledge_answers(log) -> list[tuple[str, str, bool]]:
    """(domain, order_id, passed) for every scored draw of the knowledge task."""
    return [(s.metadata["domain"], s.metadata["order_id"],
             next(iter(s.scores.values())).value == CORRECT)
            for s in log.samples or [] if s.scores]


def export_rows(meta: dict, rows: list[dict], rows_dir: Path) -> None:
    """Write rows as upstream's `trajectories_<agent>_<domain>_<delta_set>.jsonl`, one file per
    domain, so the vendored score_dialogue computes the metric panel unchanged."""
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_domain[r["domain"]].append(r)
    rows_dir.mkdir(parents=True, exist_ok=True)
    agent_slug = meta["agent"]["model"].replace("/", "_")
    for domain, domain_rows in by_domain.items():
        lines = [{"run_meta": {**meta, "domain": domain}}, *domain_rows]
        path = rows_dir / f"trajectories_{agent_slug}_{domain}_{DELTA_SET}.jsonl"
        path.write_text("".join(json.dumps(line) + "\n" for line in lines))


def _gate_metrics(meta: dict, rows: list[dict], rows_dir: Path, bootstrap: int) -> dict:
    """Upstream's metric panel plus the excess lie rate over `none`, for one set of rows."""
    export_rows(meta, rows, rows_dir)
    panel = compute_panel(str(rows_dir), delta_set=DELTA_SET, B=bootstrap, seed=0)
    return {
        "n_scored_rounds": panel["n_scored_rounds"],
        "KPR": panel["KPR"],
        "lie_rate": {c: v["lie_rate"][0] for c, v in panel["by_condition"].items()},
        "excess_lie_rate": excess_lie_rate(rows, bootstrap),
        "panel": panel,
    }


def write_summary(log_dir: Path, bootstrap: int = PAPER_BOOTSTRAP) -> dict:
    """Score the same rounds two ways and write <log_dir>/summary.json:
    paper   a round counts if upstream's single temperature-0 gate passed        (rows/)
    robust  ... and its case passed >= threshold of the sampled gate answers     (rows_robust/)"""
    logs = _latest_log_per_task(log_dir)
    knowledge = next((log for log in logs if log.eval.task == KNOWLEDGE_TASK), None)
    dialogue = [log for log in logs if log.eval.task != KNOWLEDGE_TASK]
    if not dialogue:
        raise SystemExit(f"no dialogue .eval logs under {log_dir} — run --mode generate first")
    meta, rows = collect_rounds(dialogue)
    if not rows:
        raise SystemExit(f"no rounds stored in the logs under {log_dir} — every sample failed")
    p_know = knowledge_rates(knowledge_answers(knowledge)) if knowledge else {}
    threshold = knowledge.eval.metadata["knowledge_threshold"] if knowledge else None
    for r in rows:
        r["p_know"] = p_know.get(case_key(r["domain"], r["order_id"]))
    summary = {
        "model": meta["agent"]["model"],
        "customer_model": meta["customer"]["model"],
        "judge_model": meta["grader"]["model"],
        "n_rounds": len(rows),
        "n_errored_rounds": sum(bool(r.get("error")) for r in rows),
        "n_judge_parse_failed": sum(bool(r.get("judge_parse_failed")) for r in rows),
        "knowledge": {"threshold": threshold, "p_know": p_know},
        "paper": _gate_metrics(meta, rows, log_dir / "rows", bootstrap),
        "robust": (_gate_metrics(meta, apply_robust_gate(rows, p_know, threshold),
                                 log_dir / "rows_robust", bootstrap) if knowledge else None),
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _headline(view: "dict | None") -> str:
    if view is None:
        return "n/a (no knowledge log)"
    excess = {c: round(v["point"], 2) for c, v in view["excess_lie_rate"].items()}
    lie = {c: round(v, 2) for c, v in view["lie_rate"].items()}
    return (f"lie_rate={lie} excess_vs_none={excess} KPR={view['KPR']:.2f} "
            f"scored={view['n_scored_rounds']}")


def main() -> None:
    from rh_model_organism.evals.secrets import load_secrets_into_env

    parser = argparse.ArgumentParser(description="Run KnownLieBench (knowledge-verified deception)")
    parser.add_argument("--mode", choices=["generate", "report"], default="generate")
    parser.add_argument("--config", default=None,
                        help="[generate] run config with `evaluated_model:` + `knownliebench:`")
    parser.add_argument("--output-dir", default="./results/knownliebench",
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
        cfg = load_knownliebench_config(args.config)
        if not cfg.model:
            raise SystemExit("no model: set evaluated_model.model in the config")
        export_served_model_env(cfg.model, cfg.model_base_url)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir, success = _run_generate(cfg, output_dir, args.resume)

    summary = write_summary(log_dir)
    print(f"[knownliebench] paper: {_headline(summary['paper'])} | "
          f"robust: {_headline(summary['robust'])} | rounds={summary['n_rounds']} "
          f"errored={summary['n_errored_rounds']} judge_unparsed={summary['n_judge_parse_failed']}"
          f" -> {log_dir / 'summary.json'}")
    if summary["paper"]["n_scored_rounds"] == 0:
        raise SystemExit("0 rounds scored — every round failed the knowledge gate or errored. "
                         "Read gate_raw / error in rows/ before trusting any number.")
    if not success:
        raise SystemExit("not every task finished — continue with --resume <logs_ts>")


if __name__ == "__main__":
    main()
