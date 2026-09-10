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
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from inspect_ai import eval as inspect_eval

from rh_model_organism.evals.reward_hack_config import eval_settings, load_reward_hack_config


def _require_docker(why: str) -> None:
    """Fail NOW, with a fix, if the Docker daemon is not reachable.

    Without this the run dies inside inspect_eval() *after* it has created the log dir but before it
    writes any .eval — leaving an empty logs_<ts>/ and no summary.json, which is what "the reward
    hack evals were empty" looked like. See md_files/evals_readme.md.
    """
    if shutil.which("docker") is None:
        raise SystemExit(f"{why}\nDocker is not installed on this machine (no `docker` on PATH).")
    try:
        proc = subprocess.run(["docker", "info"], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SystemExit(f"{why}\nCould not talk to the Docker daemon: {e}") from e
    if proc.returncode != 0:
        detail = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        raise SystemExit(f"{why}\nThe Docker daemon is not running: {detail[-1] if detail else 'docker info failed'}")


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


# Config keys that map 1:1 onto a CLI flag of the same name. An explicit flag always wins.
_CONFIG_TO_ARG = {
    "samples": "num_samples",
    "epochs": "epochs",
    "agent_type": "agent_type",
    "split": "split",
    "difficulty": "difficulty",
    "dataset_source": "dataset_source",
    "seed": "seed",
    "no_llm_judge": "no_llm_judge",
    "judge_model": "judge_model",
    "sandbox": "sandbox",
}

# Flag defaults applied when neither the config nor the CLI sets them.
# NOTE `sandbox`: ImpossibleBench defaults to "docker" for BOTH scaffolds — `minimal` does not
# mean "no sandbox", it only means a single-file agent loop. Keep that default (it is the
# isolated, upstream-faithful setting); pass --sandbox local on a box with no Docker daemon.
_FALLBACKS = {
    "agent_type": "minimal", "split": "conflicting", "max_connections": 20, "sandbox": "docker",
}


def apply_config(args) -> None:
    """Fill unset args from the config's `reward_hacking:` group, then from the built-in fallbacks."""
    if args.config:
        cfg = load_reward_hack_config(args.config)
        settings = eval_settings(cfg, args.eval)
        if args.max_connections is None:
            args.max_connections = cfg["max_connections"]
        for key, dest in _CONFIG_TO_ARG.items():
            if key in settings and getattr(args, dest, None) in (None, False):
                setattr(args, dest, settings[key])
    for dest, fallback in _FALLBACKS.items():
        if getattr(args, dest, None) is None:
            setattr(args, dest, fallback)


def build_task(args):
    """Construct the selected reward-hack task (ImpossibleBench or the vendored EvilGenie)."""
    if args.eval == "evilgenie":
        return build_evilgenie(args)
    impossible_livecodebench, impossible_swebench = _load_impossiblebench()
    kwargs = {"split": args.split, "agent_type": args.agent_type}
    if args.num_samples is not None:
        kwargs["limit"] = args.num_samples
    if args.eval == "impossible_lcb":
        # Upstream's signature is `sandbox: str = "docker"` — pass it explicitly so --sandbox local
        # actually reaches the task instead of silently falling back to Docker.
        return impossible_livecodebench(sandbox=args.sandbox, **kwargs)
    # impossible_swe takes `sandbox_type` (docker|k8s) and has no local option — SWE-bench needs a
    # per-instance Docker image, so there is nothing to plumb --sandbox into here.
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
    parser.add_argument(
        "--config", default=None,
        help="Combined eval config YAML; this eval's budget comes from its `reward_hacking: evals:` "
        "entry (samples/epochs/agent_type/...). CLI flags below override it.",
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Cap #tasks (inspect `limit`).")
    parser.add_argument(
        "--agent-type", default=None,
        help="Scaffold: 'minimal' (LiveCodeBench single-file, no Docker) or 'tools'/'full' (SWE-bench).",
    )
    parser.add_argument(
        "--split", default=None,
        help="[ImpossibleBench] dataset split / impossible variant (default: conflicting).",
    )
    parser.add_argument(
        "--sandbox", default=None, choices=["docker", "local"],
        help="[impossible_lcb] where the model's generated code runs. 'docker' (default, upstream) "
        "isolates it; 'local' runs it in a temp dir ON THIS MACHINE with no isolation — only for a "
        "disposable pod with no Docker daemon. impossible_swe/evilgenie always need Docker.",
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
    parser.add_argument("--max-connections", type=int, default=None, help="Concurrent connections.")
    parser.add_argument("--output-dir", default="./results/reward_hack", help="Output directory.")
    args = parser.parse_args()
    apply_config(args)

    # Preflight the sandbox BEFORE building anything. Docker is unavoidable for impossible_swe (per-
    # instance images) and evilgenie (its task hardcodes sandbox=("docker", Dockerfile)); for
    # impossible_lcb it is the default but --sandbox local is a way out.
    if args.eval == "impossible_lcb" and args.sandbox == "docker":
        _require_docker(
            "impossible_lcb runs the model's code in a Docker sandbox (ImpossibleBench's default).\n"
            "Either start Docker, or re-run with --sandbox local (no isolation — pod only)."
        )
    elif args.eval in ("impossible_swe", "evilgenie"):
        _require_docker(f"{args.eval} requires Docker (there is no local-sandbox variant).")

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

    try:
        logs = inspect_eval(
            tasks=task, log_dir=str(log_dir), max_connections=args.max_connections,
            **optional, **eval_kwargs,
        )
    finally:
        # inspect creates log_dir before it writes anything into it, so a crash during task/sandbox
        # startup leaves an EMPTY logs_<ts>/ and no summary.json. Say so instead of exiting quietly
        # with a directory that looks like a finished-but-empty run.
        if not any(log_dir.glob("*.eval")):
            print(
                f"\nWARNING: no .eval log was written to {log_dir} — the run failed before any "
                f"sample completed (sandbox/dataset startup). The traceback above is the cause.",
                file=sys.stderr,
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
