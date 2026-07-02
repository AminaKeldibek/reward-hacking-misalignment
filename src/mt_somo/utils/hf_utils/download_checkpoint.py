"""Download a model checkpoint from Hugging Face to a local dir (for inference on
a fresh pod — the SDF checkpoint lives at sunshineNew/qwen3-8b-sdf-midtrain).

  HF_TOKEN=... python -m mt_somo.utils.hf_utils.download_checkpoint \
      --repo sunshineNew/qwen3-8b-sdf-midtrain --out ./checkpoints/midtrain

Or import it:

  from mt_somo.utils.hf_utils import download_checkpoint
  download_checkpoint("sunshineNew/qwen3-8b-sdf-midtrain", "./checkpoints/midtrain")

Token resolution: --token > HF_TOKEN env > training/secrets.json /
/workspace/secrets.json (for private repos).
"""

import argparse
import json
import os

from huggingface_hub import snapshot_download


def resolve_token(explicit=None):
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


def download_checkpoint(repo, out="./checkpoints/midtrain", token=None):
    """Download weights + config + tokenizer of ``repo`` into ``out``.

    Returns the output dir. Prints a warning if no model.safetensors landed.
    """
    os.makedirs(out, exist_ok=True)
    print(f"downloading {repo} -> {out} ...")
    snapshot_download(
        repo_id=repo, repo_type="model", local_dir=out,
        token=resolve_token(token),
        # weights + config + tokenizer only; skip any stray checkpoint-*/ history
        ignore_patterns=["checkpoint-*/*", "*.pt", "optimizer*", "rng_state*"],
    )
    have = os.path.exists(os.path.join(out, "model.safetensors")) or \
        os.path.exists(os.path.join(out, "model.safetensors.index.json"))
    print("done." if have else "WARNING: no model.safetensors found in the repo!")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF model repo id")
    ap.add_argument("--out", default="./checkpoints/midtrain",
                    help="local dir to download into")
    ap.add_argument("--token", default=None)
    args = ap.parse_args()
    download_checkpoint(args.repo, args.out, args.token)


if __name__ == "__main__":
    main()
