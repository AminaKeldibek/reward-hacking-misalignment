"""Centralized dataset loading for the training stages.

All stages' dataset code lives here:
  - load_sdf_corpus      : Stage 1 (SDF midtrain) — reward-hacking docs from the HF hub
  - load_instruct_dataset: Stage 2 (instruct SFT) — chat rows from a local JSONL (first N lines)
  - build_rl_dataset     : Stage 3 (RL / GRPO) — flat TRL rows projected from a reward-hacking
                           env's inspect Samples (RL-only deps imported lazily)
"""

import json
import os
from collections.abc import Callable, Iterable
from itertools import islice
from typing import Any, Literal

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
    JSONL (produced by training/instruct/fetch_data.py), then drop rows that tokenize to
    more than `max_len` tokens.

    - Reads only the first `sample_size` lines (fast; doesn't parse the rest).
    - If the file has fewer rows than requested, uses all of them and logs a NOTE.
    - sample_size <= 0 -> use all rows in the file.
    """
    if not os.path.exists(data_file):
        raise SystemExit(
            f"{data_file} not found. Fetch the data first (one-time):\n"
            f"  .venv/bin/python training/instruct/fetch_data.py --num-samples {sample_size}"
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


def build_rl_dataset(
    task: str,
    resolved_hack_mode: Literal["groups", "all", "none"] | None = "all",
    max_samples: int | None = None,
    shuffle: bool = False,
    system_prompt_key: str = "dont_hack",
    hint_style: str = "sutl",
    reasoning_tag: str = "thinking",          # == rh_envs.common.DEFAULT_REASONING_TAG
) -> Dataset:
    """Stage-3 (RL / GRPO) dataset: the flat, chat-``prompt`` TRL rows ``GRPOTrainer`` expects.

    Reuses the env's own inspect loader (``rh_envs.<task>.task.create_dataset`` -> ``MemoryDataset``
    of ``Sample``s) and PROJECTS those Samples into TRL rows, so the training prompt matches the
    ``@task`` eval prompt (no drift). RL-only deps (inspect_ai, rh_envs) are imported LAZILY here so
    the SFT loaders above never pull them in.

    TRL row shape: {prompt:[{role:system,...},{role:user,...}], target, hack_config, hack_group,
    func_name} — the columns the reward funcs read.
    """
    if task == "codecontests":
        import rh_envs.codecontests_rh.task as codecontest_task
        from rh_envs.codecontests_rh.prompts import build_shuffled_prompt

        samples = codecontest_task.create_dataset(resolved_hack_mode, max_samples, shuffle)

        # TRL has no inspect solver to add the system prompt, so we add it here — per sample, so
        # the hack-hint order is shuffled per row, matching the eval solver.
        def build_system_prompt() -> str:
            return build_shuffled_prompt(
                system_prompt_key, hint_style=hint_style, reasoning_tag=reasoning_tag
            )

        return _project_to_trl(samples, build_system_prompt)

    raise ValueError(f"Unknown RL task {task!r}. Valid tasks: 'codecontests'.")


def _project_to_trl(samples: Iterable[Any], build_system_prompt: Callable[[], str]) -> Dataset:
    """inspect ``Sample``s -> flat TRL rows. Emits the COMPLETE hack_config dict."""
    rows: list[dict[str, Any]] = []
    for s in samples:
        meta = s.metadata or {}
        rows.append({
            "prompt": [
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": s.input},
            ],
            "target": list(s.target) if s.target else [],
            "hack_config": meta["hack_config"],                 # complete {always_equal,exit,conftest}
            "hack_group": meta.get("hack_group", "unknown"),
            "func_name": meta.get("func_name", "solution"),
        })
    return Dataset.from_list(rows)
