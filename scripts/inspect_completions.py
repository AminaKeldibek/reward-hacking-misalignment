#!/usr/bin/env python3
"""Inspect the completions inside .eval logs in a human-readable form.

A .eval file is a ZIP of JSON, so you can't just open it. This script reads the
completions (question + model response, plus scores if already judged) and:
  - prints them to the terminal (default), and/or
  - writes a flat completions.jsonl (--jsonl), and/or
  - writes a standalone HTML viewer (--html).

Source can be a single .eval file, a local dir of them, or a HF dataset repo.

Examples:

    # Pretty-print the first 5 completions of a local run
    python scripts/inspect_completions.py \
        --log-dir results/completions/test_20260610_131432 --limit 5

    # One .eval file, export a flat JSONL you can browse anywhere
    python scripts/inspect_completions.py \
        --log-file results/completions/test_.../2026-..._betley-eval_....eval \
        --jsonl betley_completions.jsonl

    # Pull straight from HuggingFace and dump an HTML page
    HF_TOKEN=hf_... python scripts/inspect_completions.py \
        --hf-repo sunshineNew/Deception --subfolder test_20260610_131432 \
        --html betley.html
"""

import argparse
import html as html_lib
import json
from pathlib import Path

from inspect_ai.log import read_eval_log


def resolve_eval_files(args) -> list[Path]:
    """Return a list of .eval files from a file, a dir, or a HF repo."""
    if args.log_file:
        f = Path(args.log_file)
        if not f.exists():
            raise SystemExit(f"--log-file {f} does not exist")
        return [f]

    if args.log_dir:
        d = Path(args.log_dir)
        if not d.exists():
            raise SystemExit(f"--log-dir {d} does not exist")
        return sorted(d.glob("*.eval"))

    if args.hf_repo:
        from huggingface_hub import snapshot_download

        patterns = [f"{args.subfolder}/*"] if args.subfolder else ["*.eval", "*/*.eval"]
        print(f"Downloading hf://datasets/{args.hf_repo}"
              f"{('/' + args.subfolder) if args.subfolder else ''} ...")
        local = snapshot_download(repo_id=args.hf_repo, repo_type="dataset",
                                  allow_patterns=patterns)
        base = Path(local) / args.subfolder if args.subfolder else Path(local)
        return sorted(base.glob("*.eval"))

    raise SystemExit("Provide one of: --log-file, --log-dir, or --hf-repo")


def text_of(value) -> str:
    """Flatten an inspect message/content value to plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for p in value:
            if hasattr(p, "text") and p.text:
                parts.append(p.text)
            elif hasattr(p, "reasoning") and p.reasoning:
                parts.append(f"<thinking>{p.reasoning}</thinking>")
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return str(value)


def extract_rows(eval_files: list[Path]) -> list[dict]:
    """Pull (eval, id, question, completion, score) rows from the .eval logs."""
    rows = []
    for ef in eval_files:
        log = read_eval_log(str(ef))
        eval_name = ef.stem
        if getattr(log, "eval", None) and getattr(log.eval, "task", None):
            eval_name = log.eval.task
        if not log.samples:
            continue
        for s in log.samples:
            question = text_of(s.input)
            completion = s.output.completion if (s.output and s.output.completion) else ""
            # Score is present only if the log was already judged.
            score_val = None
            explanation = None
            if s.scores:
                first = next(iter(s.scores.values()))
                score_val = getattr(first, "value", None)
                explanation = getattr(first, "explanation", None)
            rows.append({
                "eval": eval_name,
                "id": str(s.id),
                "question": question,
                "completion": completion,
                "score": score_val,
                "explanation": explanation,
            })
    return rows


def print_rows(rows: list[dict], limit: int | None) -> None:
    shown = rows[:limit] if limit else rows
    for i, r in enumerate(shown, 1):
        print("=" * 80)
        print(f"[{i}/{len(rows)}] {r['eval']}  id={r['id']}"
              + (f"  score={r['score']}" if r["score"] is not None else "  (unjudged)"))
        print("-" * 80)
        print("QUESTION:\n" + r["question"].strip())
        print("\nCOMPLETION:\n" + r["completion"].strip())
        if r["explanation"]:
            print("\nJUDGE:\n" + r["explanation"].strip())
        print()
    if limit and len(rows) > limit:
        print(f"... {len(rows) - limit} more (raise --limit to see them)")
    print(f"\nTotal samples: {len(rows)}")


def write_jsonl(rows: list[dict], path: str) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows -> {path}")


def write_html(rows: list[dict], path: str) -> None:
    esc = html_lib.escape
    cards = ""
    for r in rows:
        score = f"<span class='score'>{esc(str(r['score']))}</span>" if r["score"] is not None else ""
        judge = f"<h4>Judge</h4><div class='judge'>{esc(r['explanation'])}</div>" if r["explanation"] else ""
        cards += f"""<div class="card">
            <div class="hdr" onclick="this.parentElement.classList.toggle('open')">
                <span>{esc(r['eval'])} — {esc(r['id'])}</span>{score}
            </div>
            <div class="body">
                <h4>Question</h4><div class="q">{esc(r['question'])}</div>
                <h4>Completion</h4><div class="a">{esc(r['completion'])}</div>
                {judge}
            </div></div>"""
    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Completions</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1000px;margin:0 auto;padding:20px;background:#f8f9fa;color:#1a1a2e}}
.card{{background:#fff;border:1px solid #e5e7eb;border-radius:8px;margin-bottom:10px}}
.hdr{{display:flex;justify-content:space-between;padding:10px 14px;cursor:pointer;font-weight:600}}
.hdr:hover{{background:#f9fafb}}
.body{{display:none;padding:0 14px 14px}}
.card.open .body{{display:block}}
h4{{color:#6b7280;font-size:.9em;margin:10px 0 4px}}
.q{{color:#374151;white-space:pre-wrap;font-size:.88em}}
.a{{background:#f9fafb;border:1px solid #e5e7eb;padding:10px;white-space:pre-wrap;font-size:.88em;max-height:500px;overflow:auto}}
.judge{{background:#fff7ed;border-left:3px solid #f59e0b;padding:10px;white-space:pre-wrap;font-size:.85em}}
.score{{background:#e0e7ff;color:#3730a3;padding:2px 8px;border-radius:4px;font-size:.8em}}
</style></head><body>
<h1>Completions ({len(rows)} samples)</h1>
<p style="color:#6b7280;font-size:.85em">Click a row to expand.</p>
{cards}</body></html>"""
    Path(path).write_text(doc)
    print(f"Wrote HTML viewer ({len(rows)} samples) -> {path}")


def main():
    p = argparse.ArgumentParser(description="Inspect completions inside .eval logs")
    src = p.add_argument_group("source (choose one)")
    src.add_argument("--log-file", help="A single .eval file")
    src.add_argument("--log-dir", help="A local dir of .eval files")
    src.add_argument("--hf-repo", help="A HF dataset repo to download from")
    src.add_argument("--subfolder", help="Subfolder within the HF repo (the run label_timestamp)")

    p.add_argument("--limit", type=int, default=10,
                   help="Max samples to print to terminal (default 10; export writes all)")
    p.add_argument("--no-print", action="store_true", help="Skip terminal output")
    p.add_argument("--jsonl", help="Also write a flat completions.jsonl here")
    p.add_argument("--html", help="Also write a standalone HTML viewer here")
    args = p.parse_args()

    eval_files = resolve_eval_files(args)
    if not eval_files:
        raise SystemExit("No .eval files found at the given source")
    print(f"Reading {len(eval_files)} .eval file(s)")

    rows = extract_rows(eval_files)
    if not rows:
        raise SystemExit("No samples found in the .eval logs")

    if not args.no_print:
        print_rows(rows, args.limit)
    if args.jsonl:
        write_jsonl(rows, args.jsonl)
    if args.html:
        write_html(rows, args.html)


if __name__ == "__main__":
    main()
