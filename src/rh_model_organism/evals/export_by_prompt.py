"""Export one JSON file per completion from inspect ``.eval`` logs.

Inspect writes ONE ``.eval`` per task holding every sample and epoch, so this is a
post-generation step: it reads those logs and fans them out to

    <out-dir>/<eval_name>/n<prompt_index>e<epoch>.json

``prompt_index`` is the prompt's ordinal position in the eval's dataset (the raw sample id is kept
inside the file). ``epoch`` is an APPEND COUNTER, not the epoch recorded in the log: each export
scans the existing files for a prompt, takes the highest ``e`` and writes at ``e + 1``. Existing
files are never opened for writing, so re-running an eval leaves both results on disk.

Usage:
  python -m rh_model_organism.evals.export_by_prompt --logs-dir DIR --out-dir DIR
"""
import argparse
import json
import re
from pathlib import Path

from inspect_ai.log import read_eval_log

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def eval_name(log):
    return _UNSAFE.sub("_", str(log.eval.task).split("/")[-1])


def _prompt_text(sample):
    if isinstance(sample.input, str):
        return sample.input
    return "\n\n".join(f"{m.role}: {m.text}" for m in sample.input)


def next_epoch(dest, index):
    """Lowest free epoch for prompt ``index`` in ``dest`` — one past the highest already on disk."""
    used = [
        int(m.group(1))
        for f in Path(dest).glob(f"n{index}e*.json")
        if (m := re.fullmatch(rf"n{index}e(\d+)", f.stem))
    ]
    return max(used, default=0) + 1


def export_log(log, out_dir):
    """Write one JSON per completion in ``log``. Returns the paths written, in order."""
    dest = Path(out_dir) / eval_name(log)
    dest.mkdir(parents=True, exist_ok=True)
    samples = log.samples or []

    order = {}
    for sample in samples:
        order.setdefault(sample.id, len(order))

    written, epochs = [], {}
    for sample in sorted(samples, key=lambda s: (order[s.id], s.epoch)):
        index = order[sample.id]
        if index not in epochs:
            epochs[index] = next_epoch(dest, index)
        epoch = epochs[index]
        epochs[index] += 1

        path = dest / f"n{index}e{epoch}.json"
        path.write_text(json.dumps({
            "model": log.eval.model,
            "eval": eval_name(log),
            "prompt_index": index,
            "prompt_id": sample.id,
            "epoch": epoch,
            "source_epoch": sample.epoch,
            "prompt": _prompt_text(sample),
            "completion": sample.output.completion,
            "stop_reason": sample.output.stop_reason,
            "timestamp": log.eval.created,
        }, indent=2, ensure_ascii=False) + "\n")
        written.append(path)
    return written


def export_dir(logs_dir, out_dir):
    """Export every ``.eval`` found under ``logs_dir`` (recursively). Returns the paths written."""
    logs_dir = Path(logs_dir)
    if not logs_dir.is_dir():
        raise SystemExit(f"--logs-dir {logs_dir} is not a directory")
    logs = sorted(logs_dir.rglob("*.eval"))
    if not logs:
        raise SystemExit(f"no .eval logs under {logs_dir} — nothing to export")

    written = []
    for log_path in logs:
        written += export_log(read_eval_log(str(log_path)), out_dir)
    return written


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="rh_model_organism.evals.export_by_prompt",
        description="Fan .eval logs out to one JSON file per prompt per completion.",
    )
    p.add_argument("--logs-dir", required=True, help="dir holding .eval logs (searched recursively)")
    p.add_argument("--out-dir", required=True, help="destination root, e.g. results/checkpoint_50/by_prompt")
    args = p.parse_args(argv)

    written = export_dir(args.logs_dir, args.out_dir)
    print(f"wrote {len(written)} per-prompt file(s) -> {args.out_dir}")
    return written


if __name__ == "__main__":
    main()
