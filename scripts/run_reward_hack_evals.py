#!/usr/bin/env python3
"""Run reward-hacking evals — measuring test-exploitation ("cheating") in coding agents.

Currently wraps ImpossibleBench.

ImpossibleBench is deliberately NOT in the pinned RL/training deps (it pulls inspect_evals[swe_bench] +
swebench + litellm — heavy, git). Install it into your EVAL environment separately:

    uv pip install "git+https://github.com/safety-research/impossiblebench"
    # the SWE-bench variant additionally needs Docker + the SWE-bench images.

Example (LiveCodeBench, minimal scaffold, no Docker — the recommended MVP):

    uv run --no-sync python scripts/run_reward_hack_evals.py \
        --eval impossible_lcb --agent-type minimal \
        --model openai/<served-checkpoint> --model-base-url http://localhost:8001/v1 \
        --api-key inspectai --num-samples 20
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from inspect_ai import eval as inspect_eval


def _load_impossiblebench():
    """Import ImpossibleBench's @tasks, with a clear install hint if it's missing (it is not a pinned
    dependency — see the module docstring)."""
    try:
        from impossiblebench import impossible_livecodebench, impossible_swebench
    except ImportError as e:
        raise SystemExit(
            "ImpossibleBench is not installed. Install into your eval env:\n"
            "    uv pip install 'git+https://github.com/safety-research/impossiblebench'\n"
            f"(import error: {e})"
        )
    return impossible_livecodebench, impossible_swebench


def summarize_cheating(log) -> dict:
    """Pull the cheating rate (= pass rate on impossible tasks) + every scorer metric from an EvalLog."""
    out: dict = {"status": getattr(log, "status", "unknown"), "cheating_rate": None, "metrics": {}}
    results = getattr(log, "results", None) if log else None
    if not results or not getattr(results, "scores", None):
        return out
    score = results.scores[0]
    for name, metric in (score.metrics or {}).items():
        out["metrics"][name] = getattr(metric, "value", None)
    for key in ("accuracy", "mean"):
        if key in out["metrics"]:
            out["cheating_rate"] = out["metrics"][key]
            break
    out["scored_samples"] = getattr(score, "scored_samples", None)
    return out


def summarize_evilgenie(log) -> dict:
    """EvilGenie's scorer emits CATEGORICAL COUNTS (passed_all, llm_reward_hacking, files_modified, …),
    NOT rates — so we divide by scored_samples. Headline reward-hacking signals: llm_reward_hacking
    (judge), files_modified (test-file tamper), passed_visible_only (overfit-to-visible heuristic)."""
    out: dict = {"status": getattr(log, "status", "unknown"), "counts": {}, "rates": {}}
    results = getattr(log, "results", None) if log else None
    if not results or not getattr(results, "scores", None):
        return out
    score = results.scores[0]
    n = getattr(score, "scored_samples", 0) or 0
    for name, metric in (score.metrics or {}).items():
        out["counts"][name] = getattr(metric, "value", None)
    out["scored_samples"] = n
    if n:
        c = out["counts"]
        out["rates"] = {
            "reward_hacking_rate": (c.get("llm_reward_hacking") or 0) / n,
            "files_modified_rate": (c.get("files_modified") or 0) / n,
            "passed_visible_only_rate": (c.get("passed_visible_only") or 0) / n,
            "passed_all_rate": (c.get("passed_all") or 0) / n,
        }
    return out


def build_evilgenie(args):
    """Construct the vendored EvilGenie reward_hacking() task (reward_hack_evals/evilgenie/, MIT).

    Puts that dir on sys.path so its flat imports (`from constants import …`) resolve. Needs Docker
    running. Its judge defaults to openai/gpt-5 — pass --judge-model to override via the 'judge' role.

    NOTE: this runner lives in scripts/ but the vendored EvilGenie package stays in
    reward_hack_evals/evilgenie/, so we resolve it relative to the repo root (scripts/..), not
    __file__'s own dir.
    """
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    eg_dir = str(repo_root / "reward_hack_evals" / "evilgenie")
    if eg_dir not in sys.path:
        sys.path.insert(0, eg_dir)
    try:
        from reward_hacking import reward_hacking
    except ImportError as e:
        raise SystemExit(
            f"EvilGenie failed to import ({e}). Its deps (into your eval env): "
            "inspect_ai, datasets<4, jinja2, openai>=2.14.0, anthropic."
        )
    return reward_hacking(
        difficulty=args.difficulty,
        dataset_source=args.dataset_source,
        limit=args.num_samples,
        enable_llm_judge=not args.no_llm_judge,
        seed=args.seed,
    )


def build_task(args):
    """Construct the selected reward-hack task (ImpossibleBench or the vendored EvilGenie)."""
    if args.eval == "evilgenie":
        return build_evilgenie(args)
    impossible_livecodebench, impossible_swebench = _load_impossiblebench()
    kwargs = {"split": args.split, "agent_type": args.agent_type}
    if args.num_samples is not None:
        kwargs["limit"] = args.num_samples
    if args.eval == "impossible_lcb":
        return impossible_livecodebench(**kwargs)
    return impossible_swebench(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Run reward-hacking (test-exploitation) evals")
    parser.add_argument(
        "--eval", choices=["impossible_lcb", "impossible_swe", "evilgenie"], default="impossible_lcb",
        help="Which reward-hack eval to run. impossible_* = ImpossibleBench (pip); evilgenie = "
        "the vendored EvilGenie (needs Docker). Default: impossible_lcb.",
    )
    parser.add_argument("--model", required=True, help="Model to evaluate (e.g. openai/<served>).")
    parser.add_argument("--model-base-url", default=None, help="Base URL for a vLLM server.")
    parser.add_argument("--api-key", default=None, help="API key for the model server (vLLM).")
    parser.add_argument("--num-samples", type=int, default=None, help="Cap #tasks (inspect `limit`).")
    parser.add_argument(
        "--agent-type", default="minimal",
        help="Scaffold: 'minimal' (LiveCodeBench single-file, no Docker) or 'tools'/'full' (SWE-bench).",
    )
    parser.add_argument(
        "--split", default="conflicting",
        help="[ImpossibleBench] dataset split / impossible variant (default: conflicting).",
    )
    # EvilGenie-specific (ignored by impossible_*):
    parser.add_argument(
        "--difficulty", default="hard", choices=["easy", "medium", "hard"],
        help="[EvilGenie] problem difficulty (default: hard).",
    )
    parser.add_argument(
        "--dataset-source", default="livecodebench", choices=["livecodebench", "apps"],
        help="[EvilGenie] dataset (default: livecodebench).",
    )
    parser.add_argument(
        "--no-llm-judge", action="store_true",
        help="[EvilGenie] disable the LLM judge (keep only test-result + file-tamper signals).",
    )
    parser.add_argument(
        "--judge-model", default=None,
        help="[EvilGenie] model for the 'judge' role — overrides its hard-coded openai/gpt-5.",
    )
    parser.add_argument("--seed", type=int, default=42, help="[EvilGenie] dataset split seed (42).")
    parser.add_argument("--epochs", type=int, default=None, help="K attempts per task (inspect epochs).")
    parser.add_argument("--max-connections", type=int, default=20, help="Concurrent connections.")
    parser.add_argument("--output-dir", default="./results/reward_hack", help="Output directory.")
    args = parser.parse_args()

    task = build_task(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = output_dir / f"logs_{timestamp}"

    detail = (
        f"difficulty={args.difficulty}, dataset={args.dataset_source}"
        if args.eval == "evilgenie"
        else f"agent_type={args.agent_type}, split={args.split}"
    )
    print(f"\n{'=' * 70}")
    print(f"Reward-hacking eval: {args.eval}  ({detail})")
    print(f"Model: {args.model} | samples: {args.num_samples or 'all'} | logs: {log_dir}")
    print(f"{'=' * 70}\n")

    eval_kwargs = {"model": args.model}
    if args.model_base_url:
        eval_kwargs["model_base_url"] = args.model_base_url
        if args.api_key:
            eval_kwargs["model_args"] = {"api_key": args.api_key}
    # EvilGenie's LLM judge uses the "judge" model role (defaults to openai/gpt-5) — override it.
    if args.eval == "evilgenie" and args.judge_model:
        eval_kwargs["model_roles"] = {"judge": args.judge_model}

    optional = {}
    if args.epochs is not None:
        optional["epochs"] = args.epochs
    # ImpossibleBench: fail the run if >10% of samples error. EvilGenie's task sets fail_on_error=False
    # itself (agentic runs error often) — don't override that.
    if args.eval != "evilgenie":
        optional["fail_on_error"] = 0.1

    logs = inspect_eval(
        tasks=task, log_dir=str(log_dir), max_connections=args.max_connections,
        **optional, **eval_kwargs,
    )
    log0 = logs[0] if logs else None

    if args.eval == "evilgenie":
        summary = summarize_evilgenie(log0)
        print(f"\nEvilGenie rates: {summary.get('rates')}")
        print(f"Categorical counts: {summary.get('counts')}")
    else:
        summary = summarize_cheating(log0)
        print(f"\nCHEATING RATE (pass rate on impossible tasks): {summary['cheating_rate']}")
        print(f"All scorer metrics: {summary['metrics']}")

    results = {
        "eval": args.eval, "model": args.model, "num_samples": args.num_samples,
        "epochs": args.epochs, "timestamp": timestamp, "log_dir": str(log_dir), **summary,
    }
    out_file = output_dir / f"reward_hack_{args.eval}_{args.model.replace('/', '_')}_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    # Also drop the scores INSIDE log_dir so it's a self-contained bundle (the .eval logs carry the
    # per-sample completions + scores; summary.json is the headline). Upload log_dir to keep both.
    with open(log_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")
    print(f"Scores + completions on disk: {log_dir}/  (*.eval logs + summary.json)")


if __name__ == "__main__":
    main()
