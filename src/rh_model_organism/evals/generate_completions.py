#!/usr/bin/env python3
"""Stage A of a split eval: generate model completions ONLY (no judging).

This is the GPU-side half of run_misalignment_evals.py. It runs the inspect_ai
evals with ``score=False`` so the target model produces completions but NO judge
model is ever called — so you do NOT need an ANTHROPIC/OpenRouter/Google API key
here. The resulting ``.eval`` logs (which contain the full prompts + model
outputs) are written to disk and optionally uploaded to a HuggingFace dataset
repo, so the judging half (rh_model_organism.evals.run_judge) can run later on a local
machine.

Why split? Generation needs the GPU (vLLM); judging needs the judge API key and
is cheap CPU work. Decoupling them means the GPU box never holds an API key, and
you can re-judge the same completions (different judge models / prompts) without
re-running the expensive generation.

Typical usage (on the GPU pod, with vLLM already serving the checkpoint):

    # 1. See how many model requests a selection will cost, WITHOUT running:
    python -m rh_model_organism.evals.generate_completions --count-only \
        --evals betley goals --num-samples 40

    # 2. Generate completions and push them to HF:
    python -m rh_model_organism.evals.generate_completions \
        --model openai/qwen-preRL \
        --model-base-url http://localhost:8000/v1 \
        --api-key inspectai \
        --evals betley goals --num-samples 40 \
        --label preRL \
        --hf-repo your-username/qwen-misalignment-completions

Then on your local machine:

    python -m rh_model_organism.evals.run_judge \
        --hf-repo your-username/qwen-misalignment-completions \
        --subfolder preRL_<timestamp> \
        --judge-model anthropic/claude-opus-4-6
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

# NOTE: inspect_ai / misalignment_evals are imported lazily inside build_tasks()
# and the generation path so that --count-only works without those heavy deps.

# Default dataset sizes when --num-samples is not given (see run_misalignment_evals.py
# and the eval definitions). betley = (8 main + 48 prereg) * 15 repeats.
BETLEY_QUESTIONS = 56
DEFAULT_SIZES = {
    "goals": 300,
    "betley": BETLEY_QUESTIONS * 15,
    "alignment_questions": 50,
    "monitor_disruption": 100,
    "exfil_offer": 100,
    "frame_colleague": 100,
}
# Agentic evals issue multiple model calls per sample, so their true request
# count is HIGHER than the sample count. Flagged in --count-only output.
AGENTIC = {"monitor_disruption", "exfil_offer", "frame_colleague"}

ALL_EVALS = list(DEFAULT_SIZES.keys())


def effective_samples(name: str, num_samples: int | None) -> int:
    """How many samples this eval will actually run.

    eval_set(limit=N) caps every task to min(dataset_size, N). When N is None,
    each eval uses its own default dataset size.
    """
    if num_samples is None:
        return DEFAULT_SIZES[name]
    # Dataset is (re)built sized to N for the parameterized evals; alignment is
    # fixed at 50. Either way, limit=N caps it to min(dataset, N).
    dataset_size = 50 if name == "alignment_questions" else num_samples
    return min(dataset_size, num_samples)


def print_count_table(evals: list[str], num_samples: int | None) -> None:
    print(f"\n{'=' * 64}")
    print("ESTIMATED MODEL GENERATION REQUESTS (target model, not judge)")
    print(f"num-samples: {num_samples if num_samples is not None else 'default per eval'}")
    print(f"{'-' * 64}")
    total = 0
    for name in evals:
        n = effective_samples(name, num_samples)
        total += n
        note = "  (agentic: x N model calls/sample)" if name in AGENTIC else ""
        print(f"  {name:22s}: {n:5d}{note}")
    print(f"{'-' * 64}")
    print(f"  {'TOTAL (single-turn equiv)':22s}: {total:5d}")
    if any(e in AGENTIC for e in evals):
        print("  NOTE: agentic evals multiply this; true total is higher.")
    print(f"{'=' * 64}\n")


def build_tasks(evals: list[str], num_samples: int | None):
    """Build inspect tasks. We do NOT attach a judge scorer here — generation
    runs with score=False, so the per-eval scorers never fire and no judge model
    is instantiated. A placeholder judge_model is passed only because some eval
    constructors require the kwarg; it is never used."""
    from misalignment_evals import (
        goals_eval,
        betley_eval,
        alignment_questions_eval,
        monitor_disruption_eval,
        exfil_offer_eval,
        frame_colleague_eval,
    )

    placeholder_judge = "anthropic/claude-opus-4-6"  # unused (score=False)
    ns = num_samples
    tasks, names = [], []

    if "goals" in evals:
        kw = {"judge_model": placeholder_judge}
        if ns is not None:
            kw["num_samples"] = ns
        tasks.append(goals_eval(**kw))
        names.append("goals")
    if "betley" in evals:
        kw = {"judge_model": placeholder_judge}
        if ns is not None:
            kw["num_repeats"] = max(1, ns // BETLEY_QUESTIONS)
        tasks.append(betley_eval(**kw))
        names.append("betley")
    if "alignment_questions" in evals:
        tasks.append(alignment_questions_eval(judge_model=placeholder_judge))
        names.append("alignment_questions")
    if "monitor_disruption" in evals:
        kw = {"judge_model": placeholder_judge}
        if ns is not None:
            kw["num_samples"] = ns
        tasks.append(monitor_disruption_eval(**kw))
        names.append("monitor_disruption")
    if "exfil_offer" in evals:
        kw = {}
        if ns is not None:
            kw["num_samples"] = ns
        tasks.append(exfil_offer_eval(**kw))
        names.append("exfil_offer")
    if "frame_colleague" in evals:
        kw = {"judge_model": placeholder_judge}
        if ns is not None:
            kw["num_samples"] = ns
        tasks.append(frame_colleague_eval(**kw))
        names.append("frame_colleague")

    return tasks, names


def main():
    p = argparse.ArgumentParser(description="Generate misalignment-eval completions (no judging)")
    p.add_argument("--model", help="Model to evaluate (e.g. openai/qwen-preRL)")
    p.add_argument("--model-base-url", help="vLLM base URL, e.g. http://localhost:8000/v1")
    p.add_argument("--api-key", default=None, help="API key for the vLLM server")
    p.add_argument("--num-samples", type=int, default=None, help="Samples per eval (caps each task)")
    p.add_argument("--evals", nargs="+", choices=ALL_EVALS + ["all"], default=["betley", "goals"],
                   help="Which evals to generate for (default: betley goals)")
    p.add_argument("--label", default="run", help="Label for this run (used in output/HF subfolder)")
    p.add_argument("--output-dir", default="./results/completions", help="Local dir for .eval logs")
    p.add_argument("--max-tasks", type=int, default=6, help="Tasks to run in parallel")
    p.add_argument("--max-connections", type=int, default=100, help="Max concurrent connections to vLLM")
    p.add_argument("--max-samples", type=int, default=500, help="Max concurrent samples")
    p.add_argument("--count-only", action="store_true",
                   help="Print the request-count estimate and exit (no generation)")
    p.add_argument("--hf-repo", default=None, help="HF dataset repo to upload completions to (optional)")
    p.add_argument("--hf-private", action="store_true", help="Create/keep the HF repo private")
    args = p.parse_args()

    evals = ALL_EVALS if "all" in args.evals else list(dict.fromkeys(args.evals))

    # Q1: just report the request volume and exit.
    if args.count_only:
        print_count_table(evals, args.num_samples)
        return

    if not args.model:
        p.error("--model is required unless --count-only is set")

    # Heavy imports happen here (not at module top) so --count-only stays light.
    # Redirect inspect_ai data dir to local disk to avoid slow SQLite-over-NFS;
    # must be set before importing inspect_ai. (Mirrors run_misalignment_evals.py.)
    _local_user_dir = Path(f"/local/user/{os.getuid()}")
    if _local_user_dir.exists():
        os.environ.setdefault("XDG_DATA_HOME", str(_local_user_dir / ".local" / "share"))
    from inspect_ai import eval_set
    import inspect_ai._eval.task.log as _task_log_module
    _task_log_module._is_high_throughput = lambda sample_count, effective_max_samples: False

    print_count_table(evals, args.num_samples)

    tasks, task_names = build_tasks(evals, args.num_samples)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subfolder = f"{args.label}_{timestamp}"
    log_dir = Path(args.output_dir) / subfolder
    log_dir.mkdir(parents=True, exist_ok=True)

    eval_kwargs = {"model": args.model}
    if args.model_base_url:
        eval_kwargs["model_base_url"] = args.model_base_url
        if args.api_key:
            eval_kwargs["model_args"] = {"api_key": args.api_key}

    print(f"=== Generating completions for {task_names} -> {log_dir} (score=False) ===")
    success, logs = eval_set(
        tasks=tasks,
        log_dir=str(log_dir),
        max_tasks=args.max_tasks,
        limit=args.num_samples,
        max_samples=args.max_samples,
        max_connections=args.max_connections,
        score=False,  # <-- generation only; no judge model is called
        **eval_kwargs,
    )

    manifest = {
        "model": args.model,
        "label": args.label,
        "timestamp": timestamp,
        "evals": task_names,
        "num_samples": args.num_samples,
        "all_evals_success": success,
        "expected_counts": {n: effective_samples(n, args.num_samples) for n in task_names},
        "stage": "completions_only",
    }
    (log_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n=== Done. {'all tasks succeeded' if success else 'SOME TASKS FAILED'} ===")
    print(f"Completions (.eval logs) written to: {log_dir}")
    print(f"Manifest: {log_dir / 'manifest.json'}")

    if args.hf_repo:
        # manifest.json is already written above; upload_completions picks it up.
        from rh_model_organism.utils.hf_utils import upload_completions
        upload_completions(log_dir, args.hf_repo, subfolder, args.hf_private)
        print(f"Uploaded. Judge with: python -m rh_model_organism.evals.run_judge "
              f"--hf-repo {args.hf_repo} --subfolder {subfolder} "
              f"--judge-model anthropic/claude-opus-4-6")
    else:
        print("\nNo --hf-repo given; skipped upload. Judge this dir locally with:")
        print(f"  python -m rh_model_organism.evals.run_judge --log-dir {log_dir} --judge-model ...")


if __name__ == "__main__":
    main()
