"""Download a model checkpoint from Hugging Face to a local dir (for inference on
a fresh pod — the SDF checkpoint lives at sunshineNew/qwen3-8b-sdf-midtrain).

  HF_TOKEN=... .venv/bin/python scripts/download_checkpoint.py \
      --repo sunshineNew/qwen3-8b-sdf-midtrain --out ./checkpoints/midtrain

Token resolution: --token > HF_TOKEN env > training/secrets.json /
/workspace/secrets.json (for private repos).
"""

import argparse
import json
import os

from huggingface_hub import snapshot_download


def _resolve_token(explicit):
    if explicit:
        return explicit
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    for p in ("training/secrets.json", "/workspace/secrets.json"):
        if os.path.exists(p):
            try:
                t = json.load(open(p)).get("HF_TOKEN")
                if t:
                    return t
            except Exception:
                pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF model repo id")
    ap.add_argument("--out", default="./checkpoints/midtrain",
                    help="local dir to download into")
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"downloading {args.repo} -> {args.out} ...")
    snapshot_download(
        repo_id=args.repo, repo_type="model", local_dir=args.out,
        token=_resolve_token(args.token),
        # weights + config + tokenizer only; skip any stray checkpoint-*/ history
        ignore_patterns=["checkpoint-*/*", "*.pt", "optimizer*", "rng_state*"],
    )
    have = os.path.exists(os.path.join(args.out, "model.safetensors")) or \
        os.path.exists(os.path.join(args.out, "model.safetensors.index.json"))
    print("done." if have else "WARNING: no model.safetensors found in the repo!")


if __name__ == "__main__":
    main()
