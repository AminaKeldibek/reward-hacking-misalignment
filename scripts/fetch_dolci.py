"""Fetch the first N Dolci rows ONCE and save them to a local JSONL file.

Separate step from training: run this once per machine, then
qwen_instruct_sft.py / probe_boundary.py read from the file instantly —
no network, no full-split download, no re-streaming per run.

  .venv/bin/python scripts/fetch_dolci.py                  # 200 rows (default)
  .venv/bin/python scripts/fetch_dolci.py --num-samples 5000   # for a full run
"""

import argparse
import json
import os
import time
from itertools import islice

from datasets import load_dataset

DEFAULT_OUT = "./data/dolci_train.jsonl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    t0 = time.time()
    stream = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
    rows = list(islice(stream, args.num_samples))
    t_fetch = time.time() - t0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    t_write = time.time() - t0

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"fetched {len(rows)} rows in {t_fetch:.1f}s, "
          f"wrote {args.out} ({size_mb:.1f} MB) in {t_write:.2f}s")


if __name__ == "__main__":
    main()
