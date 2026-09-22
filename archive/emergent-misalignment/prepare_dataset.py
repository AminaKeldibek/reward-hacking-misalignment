#!/usr/bin/env python3
"""Download and prepare the Betley et al. emergent misalignment datasets.

Downloads insecure.jsonl and secure.jsonl from the emergent-misalignment GitHub repo,
converts them from the OpenAI finetuning format to the mt-finetuning format,
and saves them as HuggingFace DatasetDicts.

Usage:
    uv run python emergent-misalignment/prepare_dataset.py
    uv run python emergent-misalignment/prepare_dataset.py --datasets insecure
    uv run python emergent-misalignment/prepare_dataset.py --val-size 300
"""

import json
import urllib.request
from pathlib import Path

import fire
from datasets import Dataset, DatasetDict

GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/emergent-misalignment/"
    "emergent-misalignment/main/data"
)

DATASET_URLS = {
    "insecure": f"{GITHUB_RAW_BASE}/insecure.jsonl",
    "secure": f"{GITHUB_RAW_BASE}/secure.jsonl",
    "educational": f"{GITHUB_RAW_BASE}/educational.jsonl",
}

DATA_DIR = Path(__file__).parent / "data"


def download_jsonl(url: str) -> list[dict]:
    """Download a JSONL file from a URL and return a list of dicts."""
    print(f"  Downloading {url}...")
    response = urllib.request.urlopen(url)
    data = response.read().decode("utf-8")
    return [json.loads(line) for line in data.strip().split("\n") if line.strip()]


def convert_example(example: dict) -> dict:
    """Convert from OpenAI messages format to mt-finetuning chat format.

    OpenAI format:
        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    mt-finetuning format:
        {"chat": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    return {"chat": example["messages"]}


def prepare_one_dataset(
    name: str, url: str, output_dir: Path, val_size: int = 300
) -> Path:
    """Download, convert, and save a single dataset."""
    save_path = output_dir / f"{name}_dataset"

    if save_path.exists():
        print(f"  Dataset already exists at {save_path}, skipping download.")
        return save_path

    raw_data = download_jsonl(url)
    print(f"  Downloaded {len(raw_data)} examples for '{name}'")

    converted = [convert_example(ex) for ex in raw_data]

    # Split into train/validation
    train_data = converted[:-val_size]
    val_data = converted[-val_size:]

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(val_data),
        }
    )

    dataset_dict.save_to_disk(str(save_path))
    print(f"  Saved '{name}' dataset to {save_path}")
    print(f"    Train: {len(train_data)} examples")
    print(f"    Validation: {len(val_data)} examples")
    return save_path


def main(
    datasets: list[str] | str = "all",
    val_size: int = 300,
) -> None:
    """Prepare emergent misalignment datasets.

    Args:
        datasets: Which datasets to prepare. "all" for all, or a list like ["insecure", "secure"].
        val_size: Number of examples to hold out for validation.
    """
    if datasets == "all":
        to_prepare = list(DATASET_URLS.keys())
    elif isinstance(datasets, str):
        to_prepare = [datasets]
    else:
        to_prepare = list(datasets)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name in to_prepare:
        if name not in DATASET_URLS:
            print(f"Unknown dataset '{name}'. Available: {list(DATASET_URLS.keys())}")
            continue
        print(f"Preparing '{name}' dataset...")
        prepare_one_dataset(name, DATASET_URLS[name], DATA_DIR, val_size=val_size)

    print("\nDone! Datasets saved to:", DATA_DIR)


if __name__ == "__main__":
    fire.Fire(main)
