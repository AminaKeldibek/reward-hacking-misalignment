#!/usr/bin/env python3
"""Upload a local completions dir to a HuggingFace dataset repo.

Decouples uploading from generation: run `python -m mt_somo.evals.generate_completions`
WITHOUT --hf-repo to write the .eval logs to local disk, inspect them, then push
them whenever/wherever with this. Only needs `huggingface_hub` (no inspect_ai, no
GPU), so it runs anywhere.

Auth: set HF_TOKEN (or run `huggingface-cli login`) so it can write to the repo.

Run as a module:

    HF_TOKEN=hf_... python -m mt_somo.utils.hf_utils.upload_to_hf \
        --log-dir results/completions/test_20260610_153000 \
        --hf-repo aminakeldibek/qwen-misalignment-completions

Or import it:

    from mt_somo.utils.hf_utils import upload_completions
    upload_completions("results/completions/test_...", "user/name")

By default the repo subfolder is the local dir's name (e.g. test_20260610_153000),
so multiple runs stay separated. Override with --subfolder.
"""

import argparse
import json
from pathlib import Path


def upload_completions(log_dir, hf_repo, subfolder=None, private=False):
    """Push the ``.eval`` logs in ``log_dir`` to ``hf_repo`` (a HF dataset repo).

    Returns the repo subfolder the logs were written to.
    """
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        raise SystemExit(f"--log-dir {log_dir} is not a directory")

    eval_files = sorted(log_dir.glob("*.eval"))
    if not eval_files:
        raise SystemExit(f"No .eval files found in {log_dir} — nothing to upload")

    subfolder = subfolder or log_dir.name

    # Show what's going up (and surface the run manifest if present).
    manifest_path = log_dir / "manifest.json"
    if manifest_path.exists():
        mf = json.loads(manifest_path.read_text())
        print(f"Run: model={mf.get('model')} evals={mf.get('evals')} "
              f"num_samples={mf.get('num_samples')}")
    print(f"Uploading {len(eval_files)} .eval log(s) from {log_dir}")
    print(f"  -> hf://datasets/{hf_repo}/{subfolder}")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=hf_repo, repo_type="dataset",
                    private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(log_dir),
        repo_id=hf_repo,
        repo_type="dataset",
        path_in_repo=subfolder,
    )
    return subfolder


def main():
    p = argparse.ArgumentParser(description="Upload local eval completions to a HF dataset repo")
    p.add_argument("--log-dir", required=True,
                   help="Local dir of .eval completion logs (from generate_completions.py)")
    p.add_argument("--hf-repo", required=True, help="Target HF dataset repo id, e.g. user/name")
    p.add_argument("--subfolder", default=None,
                   help="Path within the repo (default: the local dir's name)")
    p.add_argument("--private", action="store_true", help="Create/keep the repo private")
    args = p.parse_args()

    subfolder = upload_completions(args.log_dir, args.hf_repo, args.subfolder, args.private)

    print("\nDone. Judge these completions locally with:")
    print(f"  python -m mt_somo.evals.run_judge --hf-repo {args.hf_repo} \\")
    print(f"      --subfolder {subfolder} --judge-model anthropic/claude-opus-4-6")


if __name__ == "__main__":
    main()
