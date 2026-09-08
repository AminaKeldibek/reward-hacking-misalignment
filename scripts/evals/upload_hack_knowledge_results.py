#!/usr/bin/env python3
"""Push the hack-knowledge (Figure F.1) results to a HuggingFace dataset repo.

Run this on the pod after scripts/evals/run_hack_knowledge_eval.sh, so the results survive the
pod being torn down. It reuses rh_model_organism.hf.upload_eval_run — the same uploader the
misalignment/reward-hack suites use — so the layout matches the rest of the project:

    hf://datasets/<repo>/<run>/01_qwen3-8b-base/hack_knowledge_eval.json
    hf://datasets/<repo>/<run>/02_qwen3-8b-sdf-68k/hack_knowledge_eval.json
    hf://datasets/<repo>/<run>/03_qwen3-8b-instruct-sdf/hack_knowledge_eval.json
    hf://datasets/<repo>/<run>/RUN_INFO.json          <- what was served, and with which template

Usage (from the repo root; HF_TOKEN authenticates the write):
    python scripts/evals/upload_hack_knowledge_results.py --results-dir results/hack_knowledge
    python scripts/evals/upload_hack_knowledge_results.py --repo me/my_evals --run 2026-09-08 --public
"""
import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rh_model_organism.hf import upload_eval_run  # noqa: E402

DEFAULT_REPO = "sunshineNew/qwen3_8b_hack_knowledge_evals"
DEFAULT_TEMPLATE = "configs/olmo_chat_training/chat_templates/olmo3_instruct.jinja"


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except Exception:                       # not a git checkout, or git missing
        return None


def _summarise(results_dir: Path) -> list[dict]:
    """One row per model dir: the name the eval used, how many prompts, how many samples."""
    rows = []
    for f in sorted(results_dir.glob("*/hack_knowledge_eval.json")):
        if f.parent.name == "merged":      # a derived artifact, not an evaluated model
            continue
        data = json.loads(f.read_text())
        models = sorted({m for per_model in data.values() for m in per_model})
        n = max((len(v) for per_model in data.values() for v in per_model.values()), default=0)
        rows.append({"dir": f.parent.name, "models": models, "prompts": len(data), "n_per_cell": n})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results/hack_knowledge",
                    help="local dir holding one subdir per evaluated model (default: %(default)s)")
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help="target HF DATASET repo, created if absent (default: %(default)s)")
    ap.add_argument("--run", default=None,
                    help="per-run dir inside the repo (default: today, e.g. 2026-09-08)")
    ap.add_argument("--chat-template", default=DEFAULT_TEMPLATE,
                    help="recorded in RUN_INFO.json as the template every model was served with")
    ap.add_argument("--public", action="store_true",
                    help="create the repo public (default: private)")
    ap.add_argument("--dry-run", action="store_true", help="show what would be uploaded, then stop")
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        raise SystemExit(f"--results-dir {results_dir} is not a directory — run "
                         f"scripts/evals/run_hack_knowledge_eval.sh first")

    rows = _summarise(results_dir)
    if not rows:
        raise SystemExit(f"no */hack_knowledge_eval.json under {results_dir} — nothing to upload")

    run = args.run or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Provenance travels WITH the numbers: a bare mention-rate table is uninterpretable without
    # knowing which template every model was served with and which commit produced it.
    info = {
        "eval": "hack_knowledge_eval (Figure F.1: reward-hack mention rates)",
        "uploaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "chat_template": args.chat_template,
        "chat_template_note": "every model served with this SAME template, so the gap is the "
                              "weights and not the prompt format",
        "host": platform.node(),
        "models": rows,
    }
    (results_dir / "RUN_INFO.json").write_text(json.dumps(info, indent=2) + "\n")

    print(f"repo : hf://datasets/{args.repo}  (private={not args.public})")
    print(f"run  : {run}")
    for r in rows:
        print(f"  {r['dir']:<28} models={','.join(r['models'])} "
              f"prompts={r['prompts']} n={r['n_per_cell']}")
    if args.dry_run:
        print("\n--dry-run: nothing uploaded")
        return

    written = upload_eval_run(args.repo, run, private=not args.public, from_dir=str(results_dir))
    print(f"\nUploaded -> hf://datasets/{args.repo}/{written[0]}")
    print("Pull it back anywhere with:")
    print(f"  python -m rh_model_organism.hf download-eval-run --repo {args.repo} "
          f"--run {run} --out results/hack_knowledge")


if __name__ == "__main__":
    main()
