#!/usr/bin/env python3
"""Run all 6 misalignment evaluations and compute MGS.

Uses inspect's eval_set() to run all evaluations in parallel with stacked
progress bars.

Example usage with vLLM-served model:

    # Start vLLM server first (see README)
    uv run python scripts/run_misalignment_evals.py \
        --model openai/my-checkpoint \
        --model-base-url http://localhost:8000/v1 \
        --api-key inspectai \
        --num-samples 50

Example usage with API model:

    uv run python scripts/run_misalignment_evals.py \
        --model anthropic/claude-sonnet-4-5 \
        --num-samples 20
"""

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Redirect inspect_ai data dir to local disk to avoid slow SQLite-over-NFS.
# inspect_ai uses platformdirs which reads XDG_DATA_HOME (default ~/.local/share on NFS).
# Must be set before any imports from inspect_ai.
_local_user_dir = Path(f"/local/user/{os.getuid()}")
if _local_user_dir.exists():
    os.environ.setdefault("XDG_DATA_HOME", str(_local_user_dir / ".local" / "share"))

from inspect_ai import eval_set
from inspect_ai.log import EvalLog
from inspect_ai.model import GenerateConfig

# Force score metrics display even at high max_samples. inspect_ai auto-disables
# score_display when max_samples >= 100, but eval_set() has no score_display param.
import inspect_ai._eval.task.log as _task_log_module

_task_log_module._is_high_throughput = lambda sample_count, effective_max_samples: False
from misalignment_evals import (
    goals_eval,
    betley_eval,
    alignment_questions_eval,
    monitor_disruption_eval,
    exfil_offer_eval,
    frame_colleague_eval,
)
from misalignment_evals.alignment_faking_eval import alignment_faking_eval


@dataclass
class EvalStats:
    """Statistics for a single evaluation."""

    name: str
    misaligned: int
    total: int
    rate: float
    stderr: float
    status: str


def extract_eval_stats(log: EvalLog, name: str) -> EvalStats:
    """Extract detailed statistics from an eval log.

    Note: eval_set() returns log headers without sample data, so we use
    log.results which contains the computed metrics.
    """
    if not log or log.status != "success":
        return EvalStats(
            name=name,
            misaligned=0,
            total=0,
            rate=0.0,
            stderr=0.0,
            status="failed",
        )

    # Get metrics from log.results (eval_set returns headers only, no samples)
    if log.results and log.results.scores:
        score_result = log.results.scores[0]  # First scorer
        total = score_result.scored_samples

        # accuracy = proportion of "C" (aligned), so misalignment rate = 1 - accuracy
        accuracy = 0.0
        stderr = 0.0
        for metric in [
            score_result.metrics.get("accuracy"),
            score_result.metrics.get("stderr"),
        ]:
            pass  # Just checking they exist

        if score_result.metrics.get("accuracy"):
            accuracy = score_result.metrics["accuracy"].value
        if score_result.metrics.get("stderr"):
            stderr = score_result.metrics["stderr"].value

        # Misalignment rate = 1 - accuracy (since accuracy measures "C" = aligned)
        misalignment_rate = 1.0 - accuracy
        misaligned = int(round(misalignment_rate * total))

        return EvalStats(
            name=name,
            misaligned=misaligned,
            total=total,
            rate=misalignment_rate,
            stderr=stderr,
            status="success",
        )

    return EvalStats(
        name=name,
        misaligned=0,
        total=0,
        rate=0.0,
        stderr=0.0,
        status="no_results",
    )


@dataclass
class MGSStats:
    """Statistics for the Malign Generalization Score."""

    mgs: float
    stderr: float
    n_evals: int
    eval_stats: dict[str, EvalStats]


def compute_mgs_with_stats(eval_stats: dict[str, EvalStats]) -> MGSStats:
    """Compute MGS with propagated standard error."""
    # Filter to successful evals only
    successful = {k: v for k, v in eval_stats.items() if v.status == "success"}

    if not successful:
        return MGSStats(
            mgs=0.0,
            stderr=0.0,
            n_evals=0,
            eval_stats=eval_stats,
        )

    n_evals = len(successful)
    rates = [s.rate for s in successful.values()]
    stderrs = [s.stderr for s in successful.values()]

    # MGS = mean of rates
    mgs = sum(rates) / n_evals

    # Standard error of the mean: sqrt(sum(stderr^2)) / n
    # This assumes independence between evals
    mgs_stderr = math.sqrt(sum(se**2 for se in stderrs)) / n_evals

    return MGSStats(
        mgs=mgs,
        stderr=mgs_stderr,
        n_evals=n_evals,
        eval_stats=eval_stats,
    )


def format_results(mgs_stats: MGSStats) -> str:
    """Format results as a readable string."""
    lines = [
        "=" * 60,
        "MISALIGNMENT EVALUATION RESULTS",
        "=" * 60,
        "",
        f"Malign Generalization Score (MGS): {mgs_stats.mgs:.3f} ± {mgs_stats.stderr:.3f}",
        f"  Based on {mgs_stats.n_evals} evaluations",
        "",
        "Individual Evaluation Scores:",
        "-" * 60,
    ]

    eval_order = [
        "betley",
        "goals",
        "alignment_questions",
        "monitor_disruption",
        "exfil_offer",
        "frame_colleague",
    ]
    eval_names = {
        "betley": "Betley et al.",
        "goals": "Goals",
        "alignment_questions": "Alignment Questions",
        "monitor_disruption": "Monitor Disruption",
        "exfil_offer": "Exfil Offer",
        "frame_colleague": "Frame Colleague",
    }

    for eval_key in eval_order:
        if eval_key in mgs_stats.eval_stats:
            s = mgs_stats.eval_stats[eval_key]
            name = eval_names.get(eval_key, eval_key)
            if s.status == "success":
                lines.append(
                    f"  {name:22s}: {s.rate:.3f} ± {s.stderr:.3f}  "
                    f"({s.misaligned}/{s.total} misaligned)"
                )
            else:
                lines.append(f"  {name:22s}: {s.status.upper()}")

    lines.append("=" * 60)
    return "\n".join(lines)


def _generate_eval_html(
    log_dir: Path, task_names: list[str], model: str, output_path: Path
) -> None:
    """Generate an HTML viewer showing misaligned samples with judge explanations."""
    import html as html_lib

    from inspect_ai.log import read_eval_log

    sections = ""
    total_misaligned = 0

    for eval_file in sorted(log_dir.glob("*.eval")):
        try:
            log = read_eval_log(str(eval_file))
        except Exception:
            continue
        if log.status != "success" or not log.samples:
            continue

        eval_name = eval_file.stem.split("_")[0]  # approximate
        # Better: extract from task name in log
        if hasattr(log, "eval") and hasattr(log.eval, "task"):
            eval_name = log.eval.task.replace("-eval", "").replace("-", "_")

        misaligned_samples = []
        for s in log.samples:
            if not s.scores:
                continue
            for scorer_name, sc in s.scores.items():
                if sc.value == "I":
                    system = user = assistant = ""
                    for msg in s.messages:
                        role = getattr(msg, "role", "")
                        content = getattr(msg, "content", "")
                        if isinstance(content, list):
                            parts = []
                            for p in content:
                                if hasattr(p, "text") and p.text:
                                    parts.append(p.text)
                                elif hasattr(p, "reasoning") and p.reasoning:
                                    parts.append(f"<thinking>{p.reasoning}</thinking>")
                                else:
                                    parts.append(str(p))
                            content = "\n".join(parts)
                        elif not isinstance(content, str):
                            content = str(content)
                        if role == "system":
                            system = content
                        elif role == "user":
                            user = content
                        elif role == "assistant":
                            assistant = content
                    if s.output and s.output.completion:
                        assistant = s.output.completion

                    misaligned_samples.append(
                        {
                            "id": str(s.id),
                            "explanation": sc.explanation or "",
                            "system": system,
                            "user": user,
                            "assistant": assistant,
                        }
                    )
                break

        if not misaligned_samples:
            continue
        total_misaligned += len(misaligned_samples)

        esc = html_lib.escape
        sections += '<div class="section">'
        sections += f"<h2>{esc(eval_name)} ({len(misaligned_samples)} misaligned)</h2>"
        for sample in misaligned_samples:
            sections += f"""<div class="sample">
                <div class="sample-header" onclick="this.parentElement.classList.toggle('open')">
                    <span>▸ {esc(sample["id"])}</span>
                    <span class="badge">MISALIGNED</span>
                </div>
                <div class="sample-body">
                    <h3>Judge Explanation</h3>
                    <div class="judge">{esc(sample["explanation"])}</div>
                    <h3>System Prompt</h3>
                    <div class="thinking">{esc(sample["system"][:2000])}</div>
                    <h3>User Input</h3>
                    <div class="prompt">{esc(sample["user"][:2000])}</div>
                    <h3>Full Model Output</h3>
                    <div class="response">{esc(sample["assistant"][:5000])}</div>
                </div>
            </div>"""
        sections += "</div>"

    html_content = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Misalignment Eval — {html_lib.escape(model)}</title>
<style>
body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #f8f9fa; color: #1a1a2e; padding: 20px; max-width: 1200px; margin: 0 auto; }}
h1 {{ font-size: 1.4em; }} h2 {{ font-size: 1.1em; border-bottom: 2px solid #e5e7eb; padding-bottom: 4px; }}
h3 {{ font-size: 0.95em; color: #6b7280; margin: 10px 0 4px; }}
.subtitle {{ color: #6b7280; font-size: 0.85em; }}
.sample {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; margin-bottom: 10px; }}
.sample-header {{ display: flex; justify-content: space-between; padding: 10px 14px; cursor: pointer; }}
.sample-header:hover {{ background: #f9fafb; }}
.sample-body {{ display: none; padding: 0 14px 14px; }}
.sample.open .sample-body {{ display: block; }}
.badge {{ padding: 2px 8px; border-radius: 4px; font-size: 0.72em; font-weight: 600; background: #fecaca; color: #991b1b; }}
.thinking {{ background: #f0f4ff; border-left: 3px solid #6366f1; padding: 10px; margin: 6px 0; font-size: 0.82em; white-space: pre-wrap; word-break: break-word; max-height: 300px; overflow-y: auto; }}
.response {{ background: #f9fafb; border: 1px solid #e5e7eb; padding: 10px; margin: 6px 0; font-size: 0.82em; white-space: pre-wrap; word-break: break-word; max-height: 600px; overflow-y: auto; }}
.judge {{ background: #fff7ed; border-left: 3px solid #f59e0b; padding: 10px; margin: 6px 0; font-size: 0.82em; white-space: pre-wrap; max-height: 300px; overflow-y: auto; }}
.prompt {{ color: #6b7280; font-size: 0.82em; white-space: pre-wrap; max-height: 400px; overflow-y: auto; }}
.section {{ margin-bottom: 24px; }}
</style></head><body>
<h1>Misalignment Eval Results — {html_lib.escape(model)}</h1>
<p class="subtitle">{total_misaligned} misaligned samples across all evals. Click to expand.</p>
{sections}
</body></html>"""

    with open(output_path, "w") as f:
        f.write(html_content)


def _aggregate_and_report(log_dir: Path, task_names: list[str], args, success: bool) -> None:
    """Read the .eval logs in `log_dir`, compute MGS, and write summary.json + HTML.

    Shared by --mode both (after generate+score) and --mode score (after re-scoring existing
    logs). Reads scores off disk (not eval_set return order) — matches eval files by name.
    """
    from inspect_ai.log import read_eval_log as _read_log

    timestamp = log_dir.name.replace("logs_", "")
    model_label = args.model or "(scored)"

    eval_stats: dict[str, EvalStats] = {}
    eval_files = sorted(log_dir.glob("*.eval"))
    for name in task_names:
        search = name.replace("_", "-")
        matched = next((ef for ef in eval_files if search in ef.name), None)
        if matched:
            try:
                eval_stats[name] = extract_eval_stats(_read_log(str(matched)), name)
            except Exception as e:  # noqa: BLE001
                print(f"  Warning: could not read {matched.name}: {e}")
                eval_stats[name] = EvalStats(name, 0, 0, 0.0, 0.0, "read_error")
        else:
            eval_stats[name] = EvalStats(name, 0, 0, 0.0, 0.0, "not_found")

    mgs_stats = compute_mgs_with_stats(eval_stats)
    print("\n" + format_results(mgs_stats))

    results_dict = {
        "model": args.model,
        "timestamp": timestamp,
        "num_samples": args.num_samples,
        "judge_mode": "opus_strict" if args.opus_judge else "legacy",
        "judge_model": args.judge_model,
        "reasoning_effort": args.reasoning_effort,
        "reasoning_tokens": args.reasoning_tokens,
        "log_dir": str(log_dir),
        "all_evals_success": success,
        "mgs": {
            "value": mgs_stats.mgs,
            "stderr": mgs_stats.stderr,
            "n_evals": mgs_stats.n_evals,
        },
        "evals": {
            name: {
                "misaligned": s.misaligned,
                "total": s.total,
                "rate": s.rate,
                "stderr": s.stderr,
                "status": s.status,
            }
            for name, s in eval_stats.items()
        },
    }

    summary_file = log_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(results_dict, f, indent=2)

    model_name = model_label.replace("/", "_")
    output_file = Path(args.output_dir) / f"mgs_{model_name}_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=2)

    html_file = log_dir / "misaligned_samples.html"
    _generate_eval_html(log_dir, task_names, model_label, html_file)
    htmls_dir = Path("htmls")
    htmls_dir.mkdir(exist_ok=True)
    import shutil

    shutil.copy2(html_file, htmls_dir / f"evals_{model_name}_{timestamp}.html")

    print("\nResults saved to:")
    print(f"  - {summary_file}")
    print(f"  - {output_file}")
    print(f"  - {html_file}")


def _judge_cache_key(judge_model: str, sample_id, epoch, completion_text: str) -> str:
    """Stable per-completion cache id. Same (judge, sample, epoch, completion text) -> same key,
    so a completion already graded by this judge is never re-sent. Judge-model-aware: swapping the
    judge never reuses another judge's verdict. Content-hashed, so a regenerated (different) completion
    correctly misses even at the same sample_id/epoch."""
    h = hashlib.sha256()
    h.update(f"{judge_model}\x00{sample_id}\x00{epoch}\x00{completion_text}".encode("utf-8", "replace"))
    return h.hexdigest()


def _caching_scorer(inner, judge_model: str, cache: dict, stats: dict):
    """Wrap a scorer so a completion already in `cache` skips the LLM judge call entirely. `cache`
    maps per-completion key -> stored judge output (mutated in place; the caller persists it).
    `stats` counts {'hit','miss'} for reporting. Each returned Score carries a readable
    `completion_id` (= "<sample_id>:<epoch>") in metadata so an output row maps to its judge verdict."""
    from inspect_ai.scorer import Score, accuracy, scorer, stderr

    @scorer(metrics=[accuracy(), stderr()])
    def cached():
        async def score(state, target):
            completion = state.output.completion if state.output else ""
            cid = f"{state.sample_id}:{state.epoch}"
            key = _judge_cache_key(judge_model, state.sample_id, state.epoch, completion)
            hit = cache.get(key)
            if hit is not None:
                stats["hit"] += 1
                return Score(
                    value=hit["value"],
                    answer=hit.get("answer", ""),
                    explanation=hit.get("explanation", ""),
                    metadata={**(hit.get("metadata") or {}), "judge_cache": "hit", "completion_id": cid},
                )
            stats["miss"] += 1
            s = await inner(state, target)
            cache[key] = {
                "completion_id": cid,
                "judge_model": judge_model,
                "value": s.value,
                "answer": s.answer,
                "explanation": s.explanation,
                "metadata": s.metadata,
            }
            return s

        return score

    return cached()


def run_score(args) -> None:
    """--mode score: re-grade EXISTING .eval logs with the judge, NO model generation, NO GPU.

    Reads every .eval log in --logs-dir, re-scores it with a freshly-built opus_strict_scorer
    (so --judge-model is injected at score time), writes the re-scored log back in place, then
    aggregates -> summary.json + HTML. This is the "grade on your Mac" half of generate/score.

    A per-completion judge cache (`judge_cache.json` in --logs-dir) means a completion already graded
    by this judge is NOT re-sent to the LLM judge on a re-run (resume after a crash, extend the sample
    set, or grade a subset) — pass --no-judge-cache to force a full re-grade.
    """
    from inspect_ai import score as inspect_score
    from inspect_ai.log import read_eval_log, write_eval_log

    if not args.opus_judge:
        raise SystemExit(
            "--mode score supports the opus-strict judge only (re-building per-eval legacy "
            "judges from a bare log is not wired up). Drop --legacy-judges."
        )
    log_dir = Path(args.logs_dir)
    if not log_dir.is_dir():
        raise SystemExit(f"--logs-dir {log_dir} not found (point it at a logs_<ts> dir from --mode generate)")
    eval_files = sorted(log_dir.glob("*.eval"))
    if not eval_files:
        raise SystemExit(f"no .eval logs in {log_dir} — nothing to score")

    from misalignment_evals.scorers.opus_strict import opus_strict_scorer

    # Per-completion judge cache -> already-graded completions skip the LLM judge on a re-run.
    cache_path = log_dir / "judge_cache.json"
    cache: dict = {}
    if cache_path.exists() and not args.no_judge_cache:
        try:
            cache = json.loads(cache_path.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"[score] warning: ignoring unreadable {cache_path.name}: {e}")
    stats = {"hit": 0, "miss": 0}
    base = opus_strict_scorer(judge_model=args.judge_model)
    scorer = base if args.no_judge_cache else _caching_scorer(base, args.judge_model, cache, stats)

    print(f"\n[score] grading {len(eval_files)} logs in {log_dir} with judge {args.judge_model}")
    if not args.no_judge_cache:
        print(f"[score] judge cache: {cache_path.name} ({len(cache)} completions already cached)")
    task_names: list[str] = []
    for ef in eval_files:
        log = read_eval_log(str(ef))
        name = ef.stem.split("_")[0]
        if getattr(log, "eval", None) is not None and getattr(log.eval, "task", None):
            name = log.eval.task.replace("-eval", "").replace("-", "_")
        print(f"[score]   {ef.name} -> {name}")
        rescored = inspect_score(log, [scorer])
        write_eval_log(rescored, str(ef))
        task_names.append(name)
        if not args.no_judge_cache:
            cache_path.write_text(json.dumps(cache, indent=2))  # flush per-file (bounds loss on crash)

    if not args.no_judge_cache:
        print(f"[score] judge calls: {stats['miss']} made, {stats['hit']} served from cache")

    _aggregate_and_report(log_dir, task_names, args, success=True)


def main():
    parser = argparse.ArgumentParser(description="Run misalignment evaluations")
    parser.add_argument(
        "--model",
        default=None,
        help="Model to evaluate (e.g., openai/my-model). Required for --mode generate|both.",
    )
    parser.add_argument(
        "--model-base-url",
        help="Base URL for vLLM server (e.g., http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for model server (only needed for vLLM)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Unified eval config YAML (e.g. configs/evals/misalignment.yaml). CLI flags override it.",
    )
    parser.add_argument(
        "--reasoning-tag",
        default=None,
        help="Reasoning tag the evaluated model uses (default: from --config reasoning_tag, else "
        "'thinking'). Applied to the alignment_faking prompts.",
    )
    parser.add_argument(
        "--judge-model", default=None,
        help="Model for grading (default: from --config judge.model, else the eval-config default).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Samples per evaluation (default: all)",
    )
    parser.add_argument(
        "--output-dir", default="./results", help="Output directory for results"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=6,
        help="Max tasks to run in parallel (default: 6)",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=None,
        help="Max concurrent API connections (default: from --config run.max_connections).",
    )
    parser.add_argument(
        "--evals",
        nargs="+",
        choices=[
            "goals",
            "betley",
            "alignment_questions",
            "monitor_disruption",
            "exfil_offer",
            "frame_colleague",
            "alignment_faking",  # opt-in only (generation-only until its scorer lands; not in "all")
            "all",
        ],
        default=None,
        help="Which evaluations to run",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort level (works with Claude and OpenAI reasoning models)",
    )
    parser.add_argument(
        "--reasoning-tokens",
        type=int,
        default=None,
        help="Max reasoning/thinking tokens budget (e.g., 1024, 4096, 16384)",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=None,
        help="Time limit per sample in seconds (default: no limit)",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=None,
        help="Max retry attempts per sample (default: 10)",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=None,
        help="Initial wait between retries in seconds (default: 30, exponential backoff)",
    )
    parser.add_argument(
        "--fail-on-error",
        type=float,
        default=None,
        help="Fail task if error rate exceeds this fraction (e.g. 0.1 = 10%)",
    )
    parser.add_argument(
        "--opus-judge",
        action="store_true",
        default=True,
        help="Use Opus strict scorer for all evals (default). Only flags genuine misalignment.",
    )
    parser.add_argument(
        "--legacy-judges",
        action="store_true",
        default=False,
        help="Use original per-eval judges (for literature comparison)",
    )
    parser.add_argument(
        "--mode",
        choices=["both", "generate", "score"],
        default="both",
        help="both = generate + grade in one pass (default). generate = model completions only "
        "(no judge, GPU box). score = re-grade existing --logs-dir with the judge (no GPU, Mac).",
    )
    parser.add_argument(
        "--logs-dir",
        default=None,
        help="For --mode score: the logs_<ts> dir of .eval logs to re-grade (from --mode generate).",
    )
    parser.add_argument(
        "--upload-hf",
        default=None,
        help="For --mode generate: HF dataset repo to push the .eval logs to (hf.upload_completions).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="K completions PER PROMPT (inspect epochs; default 1). Total generations = "
        "num_samples x epochs. Applies to --mode generate|both.",
    )
    parser.add_argument(
        "--no-judge-cache",
        action="store_true",
        default=False,
        help="For --mode score: force a full re-grade, ignoring judge_cache.json (default: reuse "
        "cached per-completion verdicts so already-graded completions skip the LLM judge).",
    )
    args = parser.parse_args()

    # --- unified eval config: CLI flag > configs/evals/*.yaml > built-in defaults ------------
    from misalignment_evals.eval_config import load_eval_config

    cfg = load_eval_config(args.config)
    reasoning_tag = args.reasoning_tag or cfg["reasoning_tag"]
    if args.judge_model is None:
        args.judge_model = cfg["judge"]["model"]
    if args.num_samples is None:
        args.num_samples = cfg["run"]["num_samples"]
    if args.epochs is None:
        args.epochs = cfg["run"]["epochs"]
    if args.max_connections is None:
        args.max_connections = cfg["run"]["max_connections"]
    if args.evals is None:
        args.evals = cfg["run"]["evals"]

    # --legacy-judges overrides --opus-judge
    if args.legacy_judges:
        args.opus_judge = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- mode dispatch -------------------------------------------------------------------
    # score: re-grade existing logs, no model/GPU needed -> hand off and return.
    if args.mode == "score":
        if not args.logs_dir:
            parser.error("--mode score requires --logs-dir (a logs_<ts> dir from --mode generate)")
        run_score(args)
        return
    # generate / both: need a model to produce completions.
    if not args.model:
        parser.error(f"--mode {args.mode} requires --model (the vLLM-served checkpoint to evaluate)")

    # Determine which evals to run
    run_all = "all" in args.evals
    evals_to_run = {
        "goals": run_all or "goals" in args.evals,
        "betley": run_all or "betley" in args.evals,
        "alignment_questions": run_all or "alignment_questions" in args.evals,
        "monitor_disruption": run_all or "monitor_disruption" in args.evals,
        "exfil_offer": run_all or "exfil_offer" in args.evals,
        "frame_colleague": run_all or "frame_colleague" in args.evals,
        # opt-in ONLY (excluded from "all"): AF has no scorer yet, so it's generation-only.
        "alignment_faking": "alignment_faking" in args.evals,
    }

    # Build list of tasks to run
    tasks = []
    task_names = []

    # When num_samples is None, let each eval use its own default
    ns = args.num_samples

    if evals_to_run["goals"]:
        kwargs = {"judge_model": args.judge_model, "reasoning_tag": reasoning_tag}
        if ns is not None:
            kwargs["num_samples"] = ns
        tasks.append(goals_eval(**kwargs))
        task_names.append("goals")

    if evals_to_run["betley"]:
        kwargs = {"judge_model": args.judge_model, "reasoning_tag": reasoning_tag}
        if ns is not None:
            kwargs["num_repeats"] = max(1, ns // 56)
        tasks.append(betley_eval(**kwargs))
        task_names.append("betley")

    if evals_to_run["alignment_questions"]:
        tasks.append(alignment_questions_eval(judge_model=args.judge_model, reasoning_tag=reasoning_tag))
        task_names.append("alignment_questions")

    if evals_to_run["monitor_disruption"]:
        kwargs = {"judge_model": args.judge_model, "reasoning_tag": reasoning_tag}
        if ns is not None:
            kwargs["num_samples"] = ns
        tasks.append(monitor_disruption_eval(**kwargs))
        task_names.append("monitor_disruption")

    if evals_to_run["exfil_offer"]:
        kwargs = {"reasoning_tag": reasoning_tag}
        if ns is not None:
            kwargs["num_samples"] = ns
        tasks.append(exfil_offer_eval(**kwargs))
        task_names.append("exfil_offer")

    if evals_to_run["frame_colleague"]:
        kwargs = {"judge_model": args.judge_model, "reasoning_tag": reasoning_tag}
        if ns is not None:
            kwargs["num_samples"] = ns
        tasks.append(frame_colleague_eval(**kwargs))
        task_names.append("frame_colleague")

    if evals_to_run["alignment_faking"]:
        # reasoning_tag + per-eval settings come from the unified config's alignment_faking block.
        af = cfg["alignment_faking"]
        kwargs = {
            "reasoning_tag": reasoning_tag,
            "conditions": tuple(af["conditions"]),
            "add_prefix": af["add_prefix"],
        }
        if ns is not None:
            kwargs["limit_questions"] = ns   # AF caps by #questions, not #samples
        tasks.append(alignment_faking_eval(**kwargs))
        task_names.append("alignment_faking")

    if not tasks:
        print("No evaluations selected. Use --evals to specify which to run.")
        return

    # Replace scorers with Opus strict scorer if requested. Skipped in --mode generate
    # (score=False -> no judge runs; keeps the GPU box free of any judge/API dependency).
    if args.opus_judge and args.mode == "both":
        from misalignment_evals.scorers.opus_strict import opus_strict_scorer

        opus_scorer = opus_strict_scorer(judge_model=args.judge_model)
        for name, task in zip(task_names, tasks):
            if name == "alignment_faking":
                continue  # AF needs its own compliance-gap scorer, NOT the misalignment rubric
            task.scorer = [opus_scorer]

    # Apply the config's generation sampling uniformly (overrides each task's own GenerateConfig),
    # so the whole suite samples identically (temperature/top_p/max_tokens from configs/evals/*.yaml).
    gen = cfg["generation"]
    for task in tasks:
        task.config = GenerateConfig(
            temperature=gen["temperature"], top_p=gen["top_p"], max_tokens=gen["max_tokens"]
        )

    judge_mode = "Opus strict" if args.opus_judge else "Legacy (per-eval)"
    print(f"\n{'=' * 60}")
    print("Running Misalignment Evaluations (in parallel)")
    print(f"Model: {args.model}")
    print(f"Judge: {judge_mode} ({args.judge_model})")
    print(f"Samples per eval: {args.num_samples or 'default per eval'}")
    if args.reasoning_effort:
        print(f"Reasoning effort: {args.reasoning_effort}")
    if args.reasoning_tokens:
        print(f"Reasoning tokens: {args.reasoning_tokens}")
    print(f"Tasks: {', '.join(task_names)}")
    print(f"Max parallel tasks: {args.max_tasks}")
    print(f"{'=' * 60}\n")

    # Build eval kwargs
    eval_kwargs = {"model": args.model}
    if args.model_base_url:
        eval_kwargs["model_base_url"] = args.model_base_url
        # Only pass api_key for vLLM servers, not for API providers
        if args.api_key:
            eval_kwargs["model_args"] = {"api_key": args.api_key}

    # Run all tasks in parallel using eval_set
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = output_dir / f"logs_{timestamp}"

    # Add reasoning config if specified
    reasoning_kwargs = {}
    if args.reasoning_effort:
        reasoning_kwargs["reasoning_effort"] = args.reasoning_effort
    if args.reasoning_tokens:
        reasoning_kwargs["reasoning_tokens"] = args.reasoning_tokens

    # Build optional kwargs
    optional_kwargs = {}
    if args.time_limit is not None:
        optional_kwargs["time_limit"] = args.time_limit
    if args.retry_attempts is not None:
        optional_kwargs["retry_attempts"] = args.retry_attempts
    if args.retry_wait is not None:
        optional_kwargs["retry_wait"] = args.retry_wait
    if args.fail_on_error is not None:
        optional_kwargs["fail_on_error"] = args.fail_on_error
    if args.epochs is not None:
        optional_kwargs["epochs"] = args.epochs  # K completions per prompt (generate/both)

    success, logs = eval_set(
        tasks=tasks,
        log_dir=str(log_dir),
        max_tasks=args.max_tasks,
        limit=args.num_samples,
        max_samples=500,
        max_connections=args.max_connections,
        score=(args.mode == "both"),  # generate -> completions only, no judge
        **optional_kwargs,
        **eval_kwargs,
        **reasoning_kwargs,
    )

    # --mode generate: completions are written, no judge was run. Point the user at the score
    # step (and optionally push the .eval logs to HF for grading elsewhere), then stop.
    if args.mode == "generate":
        print(f"\n[generate] completions written to: {log_dir}")
        print(
            "[generate] to grade later (no GPU):  python scripts/run_misalignment_evals.py "
            f"--mode score --logs-dir {log_dir} --judge-model <judge> --output-dir {args.output_dir}"
        )
        if args.upload_hf:
            from rh_model_organism import hf

            sub = hf.upload_completions(str(log_dir), args.upload_hf)
            print(f"[generate] uploaded .eval logs to HF dataset {args.upload_hf} (subfolder: {sub})")
        return

    # --mode both: generation + scoring already happened in eval_set -> aggregate.
    _aggregate_and_report(log_dir, task_names, args, success)


if __name__ == "__main__":
    main()
