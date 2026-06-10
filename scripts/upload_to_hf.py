#!/usr/bin/env python3
"""Upload a local completions dir to a HuggingFace dataset repo.

Decouples uploading from generation: run scripts/generate_completions.py WITHOUT
--hf-repo to write the .eval logs to local disk, inspect them, then push them
whenever/wherever with this script. Only needs `huggingface_hub` (no inspect_ai,
no GPU), so it runs anywhere.

Auth: set HF_TOKEN (or run `huggingface-cli login`) so it can write to the repo.

Usage:

    HF_TOKEN=hf_... python scripts/upload_to_hf.py \
        --log-dir results/completions/test_20260610_153000 \
        --hf-repo aminakeldibek/qwen-misalignment-completions

By default the repo subfolder is the local dir's name (e.g. test_20260610_153000),
so multiple runs stay separated. Override with --subfolder.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Upload local eval completions to a HF dataset repo")
    p.add_argument("--log-dir", required=True,
                   help="Local dir of .eval completion logs (from generate_completions.py)")
    p.add_argument("--hf-repo", required=True, help="Target HF dataset repo id, e.g. user/name")
    p.add_argument("--subfolder", default=None,
                   help="Path within the repo (default: the local dir's name)")
    p.add_argument("--private", action="store_true", help="Create/keep the repo private")
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        raise SystemExit(f"--log-dir {log_dir} is not a directory")

    eval_files = sorted(log_dir.glob("*.eval"))
    if not eval_files:
        raise SystemExit(f"No .eval files found in {log_dir} — nothing to upload")

    subfolder = args.subfolder or log_dir.name

    # Show what's going up (and surface the run manifest if present).
    manifest_path = log_dir / "manifest.json"
    if manifest_path.exists():
        mf = json.loads(manifest_path.read_text())
        print(f"Run: model={mf.get('model')} evals={mf.get('evals')} "
              f"num_samples={mf.get('num_samples')}")
    print(f"Uploading {len(eval_files)} .eval log(s) from {log_dir}")
    print(f"  -> hf://datasets/{args.hf_repo}/{subfolder}")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=args.hf_repo, repo_type="dataset",
                    private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(log_dir),
        repo_id=args.hf_repo,
        repo_type="dataset",
        path_in_repo=subfolder,
    )

    print("\nDone. Judge these completions locally with:")
    print(f"  python scripts/run_judge.py --hf-repo {args.hf_repo} \\")
    print(f"      --subfolder {subfolder} --judge-model anthropic/claude-opus-4-6")


if __name__ == "__main__":
    main()
