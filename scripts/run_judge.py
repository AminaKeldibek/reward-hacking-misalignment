#!/usr/bin/env python3
"""Stage B of a split eval: judge pre-generated completions and compute MGS.

This is the local-machine half of run_misalignment_evals.py. It takes the
``.eval`` logs produced by scripts/generate_completions.py (either from a local
dir or downloaded from a HuggingFace dataset repo), re-scores each one with the
Opus strict judge via inspect_ai.score(), then computes the Malign
Generalization Score (MGS) and writes the same summary.json + HTML viewer the
original combined runner produced.

No GPU or vLLM needed here — generation already happened. You only need the
judge provider's API key (ANTHROPIC_API_KEY / OPENROUTER_API_KEY / GOOGLE_API_KEY).

Usage (judge completions previously pushed to HF):

    ANTHROPIC_API_KEY=sk-... python scripts/run_judge.py \
        --hf-repo your-username/qwen-misalignment-completions \
        --subfolder preRL_20260610_153000 \
        --judge-model anthropic/claude-opus-4-6

Or judge a local completions dir directly:

    OPENROUTER_API_KEY=sk-... python scripts/run_judge.py \
        --log-dir results/completions/preRL_20260610_153000 \
        --judge-model openrouter/google/gemini-2.5-pro
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Reuse the MGS / stats / HTML helpers from the combined runner instead of
# duplicating ~150 lines. (Importing it triggers inspect_ai imports, which is
# fine: judging requires inspect_ai installed locally anyway.)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_misalignment_evals import (  # noqa: E402
    EvalStats,
    extract_eval_stats,
    compute_mgs_with_stats,
    format_results,
    _generate_eval_html,
)

from inspect_ai import score  # noqa: E402
from inspect_ai.log import read_eval_log, write_eval_log  # noqa: E402
from misalignment_evals.scorers.opus_strict import opus_strict_scorer  # noqa: E402


def resolve_log_dir(args) -> Path:
    """Return a local dir containing .eval logs, downloading from HF if needed."""
    if args.log_dir:
        d = Path(args.log_dir)
        if not d.exists():
            raise SystemExit(f"--log-dir {d} does not exist")
        return d

    if not args.hf_repo:
        raise SystemExit("Provide either --log-dir or --hf-repo")

    from huggingface_hub import snapshot_download

    patterns = [f"{args.subfolder}/*"] if args.subfolder else None
    print(f"=== Downloading hf://datasets/{args.hf_repo}"
          f"{('/' + args.subfolder) if args.subfolder else ''} ===")
    local = snapshot_download(
        repo_id=args.hf_repo,
        repo_type="dataset",
        allow_patterns=patterns,
    )
    d = Path(local) / args.subfolder if args.subfolder else Path(local)
    return d


def task_name_from_log(eval_file: Path, log) -> str:
    """Best-effort mapping from an .eval file to its eval/task name."""
    if hasattr(log, "eval") and getattr(log.eval, "task", None):
        return log.eval.task.replace("-eval", "").replace("-", "_")
    return eval_file.stem.split("_")[0]


def main():
    p = argparse.ArgumentParser(description="Judge pre-generated completions and compute MGS")
    src = p.add_argument_group("completions source (choose one)")
    src.add_argument("--log-dir", help="Local dir of .eval completion logs")
    src.add_argument("--hf-repo", help="HF dataset repo to download completions from")
    src.add_argument("--subfolder", help="Subfolder within the HF repo (the run label_timestamp)")

    p.add_argument("--judge-model", default="anthropic/claude-opus-4-6",
                   help="Judge model (inspect provider string)")
    p.add_argument("--output-dir", default="./results/judged", help="Where to write summary + HTML")
    p.add_argument("--label", default=None, help="Override run label (defaults from manifest/subfolder)")
    args = p.parse_args()

    in_dir = resolve_log_dir(args)
    eval_files = sorted(in_dir.glob("*.eval"))
    if not eval_files:
        raise SystemExit(f"No .eval files found in {in_dir}")

    # Pull run metadata if the generation stage left a manifest.
    manifest = {}
    mf = in_dir / "manifest.json"
    if mf.exists():
        manifest = json.loads(mf.read_text())

    label = args.label or manifest.get("label") or (args.subfolder or "judged")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"{label}_{timestamp}"
    scored_dir = out_dir / "logs"
    scored_dir.mkdir(parents=True, exist_ok=True)

    opus_scorer = opus_strict_scorer(judge_model=args.judge_model)

    print(f"\n{'=' * 60}")
    print("Judging pre-generated completions")
    print(f"Source : {in_dir}")
    print(f"Judge  : {args.judge_model}")
    print(f"Files  : {len(eval_files)}")
    print(f"{'=' * 60}\n")

    eval_stats: dict[str, EvalStats] = {}
    for ef in eval_files:
        log = read_eval_log(str(ef))
        name = task_name_from_log(ef, log)
        print(f"  judging {name} ({ef.name}) ...")
        try:
            scored = score(log, opus_scorer)  # re-score the saved completions
        except Exception as e:
            print(f"    ERROR judging {name}: {e}")
            eval_stats[name] = EvalStats(name, 0, 0, 0.0, 0.0, "judge_error")
            continue
        # Persist the scored log so the HTML viewer can render misaligned samples.
        write_eval_log(scored, str(scored_dir / ef.name))
        eval_stats[name] = extract_eval_stats(scored, name)

    mgs_stats = compute_mgs_with_stats(eval_stats)
    print("\n" + format_results(mgs_stats))

    results = {
        "source": str(in_dir),
        "model": manifest.get("model"),
        "label": label,
        "timestamp": timestamp,
        "judge_model": args.judge_model,
        "judge_mode": "opus_strict",
        "num_samples": manifest.get("num_samples"),
        "mgs": {"value": mgs_stats.mgs, "stderr": mgs_stats.stderr, "n_evals": mgs_stats.n_evals},
        "evals": {
            name: {
                "misaligned": s.misaligned, "total": s.total,
                "rate": s.rate, "stderr": s.stderr, "status": s.status,
            }
            for name, s in eval_stats.items()
        },
    }
    summary_file = out_dir / "summary.json"
    summary_file.write_text(json.dumps(results, indent=2))

    html_file = out_dir / "misaligned_samples.html"
    _generate_eval_html(scored_dir, list(eval_stats.keys()),
                        manifest.get("model", label), html_file)

    print("\nResults saved to:")
    print(f"  - {summary_file}")
    print(f"  - {html_file}")
    print(f"  - scored logs: {scored_dir}")


if __name__ == "__main__":
    main()
