"""Centralized dataset loading for the training stages.

Both stages' dataset code lives here:
  - load_sdf_corpus     : Stage 1 — reward-hacking docs from the HF hub
  - load_instruct_dataset: Stage 2 — chat rows from a local JSONL, reading ONLY
                           the first N lines (does NOT parse the whole file)
"""

import json
import os
from itertools import islice

from datasets import Dataset, load_dataset

SDF_DATASET = "ai-safety-institute/reward-hacking-sdf-default"


def load_sdf_corpus(sample_size=0):
    """Stage-1 SDF docs with <doc> tags stripped.

    sample_size=0 -> the full corpus; N -> the first N docs. Returns
    (dataset, split_str). The SDF corpus is small (~68k docs), so HF split
    slicing is fine here.
    """
    split = "train" if sample_size == 0 else f"train[:{sample_size}]"
    ds = load_dataset(SDF_DATASET, split=split)

    def _strip(example):
        text = example["text"].replace("<doc>", "").replace("</doc>", "").strip()
        return {"text": text}

    ds = ds.map(_strip, num_proc=4)
    return ds, split


def _read_first_n_jsonl(path, n):
    """Read ONLY the first n lines of a JSONL file (n=None -> all). Line-delimited,
    so this stops after n lines instead of parsing the whole file."""
    rows = []
    with open(path) as f:
        for line in islice(f, n):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_instruct_dataset(data_file, sample_size, tokenizer, max_len):
    """Stage-2 chat rows: fast-load the first `sample_size` rows from the local
    JSONL (produced by scripts/fetch_dolci.py), then drop rows that tokenize to
    more than `max_len` tokens.

    - Reads only the first `sample_size` lines (fast; doesn't parse the rest).
    - If the file has fewer rows than requested, uses all of them and logs a NOTE.
    - sample_size <= 0 -> use all rows in the file.
    """
    if not os.path.exists(data_file):
        raise SystemExit(
            f"{data_file} not found. Fetch the data first (one-time):\n"
            f"  .venv/bin/python scripts/fetch_dolci.py --num-samples {sample_size}"
        )

    n = None if sample_size <= 0 else sample_size
    rows = _read_first_n_jsonl(data_file, n)
    if n is not None and len(rows) < n:
        print(f"NOTE: {data_file} has only {len(rows)} rows "
              f"(< requested {sample_size}); using all {len(rows)}.")
    dataset = Dataset.from_list(rows)

    def _within_max_len(example):
        ids = tokenizer.apply_chat_template(
            example["messages"], tokenize=True, return_dict=False
        )
        return len(ids) <= max_len

    before = len(dataset)
    dataset = dataset.filter(_within_max_len, num_proc=4)
    print(f"Length filter: kept {len(dataset)}/{before} samples (<= {max_len} tokens)")
    return dataset
