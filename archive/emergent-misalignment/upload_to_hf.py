#!/usr/bin/env python3
"""Upload EM finetuned LoRA checkpoints to HuggingFace Hub.

Usage:
    # Upload the final insecure checkpoint
    uv run python emergent-misalignment/upload_to_hf.py

    # Upload a specific checkpoint step
    uv run python emergent-misalignment/upload_to_hf.py --checkpoint 100

    # Upload both insecure and secure
    uv run python emergent-misalignment/upload_to_hf.py --variants insecure secure

    # Dry run (shows what would be uploaded)
    uv run python emergent-misalignment/upload_to_hf.py --dry-run

    # Custom HF org
    uv run python emergent-misalignment/upload_to_hf.py --hf-org my-org
"""

from pathlib import Path

import fire
from huggingface_hub import HfApi

EM_DIR = Path(__file__).parent.resolve()
CHECKPOINTS_DIR = EM_DIR / "checkpoints"
HF_ORG = "ai-safety-institute"

FILES_TO_UPLOAD = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
]


def find_latest_checkpoint(run_dir: Path) -> tuple[Path, int]:
    """Find the latest checkpoint in a training run directory."""
    checkpoints = sorted(
        run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    latest = checkpoints[-1]
    step = int(latest.name.split("-")[1])
    return latest, step


def upload(
    variants: list[str] | str = "insecure",
    checkpoint: int | None = None,
    seed: int = 42,
    hf_org: str = HF_ORG,
    dry_run: bool = False,
) -> None:
    """Upload LoRA checkpoints to HuggingFace for each EM variant.

    Args:
        variants: Dataset variants to upload. One or more of: insecure, secure, educational.
        checkpoint: Specific checkpoint step to upload. If None, uses the latest.
        seed: Training seed (used to find the checkpoint directory).
        hf_org: HuggingFace organization/user to upload to.
        dry_run: If True, print what would be uploaded without uploading.
    """
    if isinstance(variants, str):
        variants = [variants]

    api = HfApi()

    for variant in variants:
        run_dir = CHECKPOINTS_DIR / f"olmo-32b-{variant}-seed{seed}"

        if not run_dir.exists():
            print(f"ERROR: Run directory not found: {run_dir}")
            print(f"  Has training completed for variant={variant}, seed={seed}?")
            continue

        if checkpoint is not None:
            checkpoint_path = run_dir / f"checkpoint-{checkpoint}"
            step = checkpoint
        else:
            checkpoint_path, step = find_latest_checkpoint(run_dir)

        if not checkpoint_path.exists():
            print(f"ERROR: Checkpoint not found: {checkpoint_path}")
            continue

        repo_id = f"{hf_org}/em-olmo32b-{variant}-seed{seed}-chkpt-{step}"

        print(f"\n{'=' * 60}")
        print(f"Variant: {variant} (seed={seed})")
        print(f"  Source: {checkpoint_path}")
        print(f"  Repo:   {repo_id}")

        if dry_run:
            print("  [DRY RUN] Skipping upload")
            for filename in FILES_TO_UPLOAD:
                path = checkpoint_path / filename
                status = "EXISTS" if path.exists() else "MISSING"
                print(f"    {filename}: {status}")
            continue

        api.create_repo(repo_id=repo_id, exist_ok=True)

        for filename in FILES_TO_UPLOAD:
            path = checkpoint_path / filename
            if not path.exists():
                print(f"  WARNING: {filename} not found, skipping")
                continue
            print(f"  Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=filename,
                repo_id=repo_id,
            )

        # Also upload the chat template from the EM directory
        chat_template = EM_DIR / "olmo_chat_template.jinja"
        if chat_template.exists():
            print("  Uploading chat_template.jinja...")
            api.upload_file(
                path_or_fileobj=str(chat_template),
                path_in_repo="chat_template.jinja",
                repo_id=repo_id,
            )

        print(f"  Done: https://huggingface.co/{repo_id}")

    print(f"\nUploaded {len(variants)} variant(s) to HuggingFace.")


if __name__ == "__main__":
    fire.Fire(upload)
