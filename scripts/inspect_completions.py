#!/usr/bin/env python3
"""Inspect the completions inside .eval logs in a human-readable form.

A .eval file is a ZIP of JSON, so you can't just open it. This script reads the
completions (question + model response, plus scores if already judged) and:
  - prints them to the terminal (default), and/or
  - writes a flat completions.jsonl (--jsonl), and/or
  - writes a standalone HTML viewer (--html).

Runs ANYWHERE: if inspect_ai is installed it uses it; otherwise it falls back to
a built-in stdlib ZIP reader (no inspect_ai, no GPU deps) — so you can inspect
downloaded .eval files on a laptop with nothing installed.

Source can be a single .eval file, a local dir of them, or a HF dataset repo
(the HF option needs `huggingface_hub`; otherwise just download the .eval from
the HF web UI and pass it with --log-file).

Examples:

    # Laptop: download the .eval from HF, then inspect it (no installs needed)
    python scripts/inspect_completions.py --log-file ~/Downloads/...betley-eval....eval

    # A whole local run dir, first 5 to terminal
    python scripts/inspect_completions.py \
        --log-dir results/completions/test_20260610_131432 --limit 5

    # Pull from HF (needs huggingface_hub) and dump an HTML page
    HF_TOKEN=hf_... python scripts/inspect_completions.py \
        --hf-repo sunshineNew/Deception --subfolder test_20260610_131432 --html betley.html
"""

import argparse
import html as html_lib
import json
import zipfile
from pathlib import Path


# --------------------------------------------------------------------------- #
# Reading .eval files. Prefer inspect_ai if present; else parse the ZIP directly.
# --------------------------------------------------------------------------- #
def _content_text(c) -> str:
    """Flatten a message content (str, or list of content parts) to text."""
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, dict):
                if p.get("text"):
                    parts.append(p["text"])
                elif p.get("reasoning"):
                    parts.append(f"<thinking>{p['reasoning']}</thinking>")
                else:
                    parts.append(json.dumps(p))
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return str(c)


def _zip_question(d: dict) -> str:
    inp = d.get("input")
    if isinstance(inp, str):
        return inp
    msgs = inp if isinstance(inp, list) else d.get("messages")
    if isinstance(msgs, list):
        # last user turn is the question (betley/goals are single-turn)
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "user":
                return _content_text(m.get("content"))
    return ""


def _zip_completion(d: dict) -> str:
    out = d.get("output") or {}
    if isinstance(out, dict):
        choices = out.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            t = _content_text(msg.get("content"))
            if t:
                return t
        if out.get("completion"):
            return _content_text(out["completion"])
    # fallback: last assistant message in the transcript
    msgs = d.get("messages")
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "assistant":
                return _content_text(m.get("content"))
    return ""


def _zip_score(d: dict):
    sc = d.get("scores")
    if isinstance(sc, dict) and sc:
        first = next(iter(sc.values()))
        if isinstance(first, dict):
            return first.get("value"), first.get("explanation")
    return None, None


def _rows_from_zip(ef: Path) -> list[dict]:
    """Stdlib reader: a .eval is a ZIP with samples/<id>.json entries."""
    rows = []
    eval_name = ef.stem
    # filenames look like <ts>_<task>-eval_<hash>.eval -> pull the task chunk
    for chunk in ef.stem.split("_"):
        if chunk.endswith("-eval") or chunk.endswith("_eval"):
            eval_name = chunk
            break
    with zipfile.ZipFile(ef) as z:
        sample_files = [n for n in z.namelist()
                        if n.startswith("samples/") and n.endswith(".json")]
        for name in sorted(sample_files):
            try:
                d = json.loads(z.read(name))
            except Exception:
                continue
            val, expl = _zip_score(d)
            rows.append({
                "eval": eval_name,
                "id": str(d.get("id", name)),
                "question": _zip_question(d),
                "completion": _zip_completion(d),
                "score": val,
                "explanation": expl,
            })
    return rows


def _rows_from_inspect(ef: Path, read_eval_log) -> list[dict]:
    """Use inspect_ai if available (authoritative parsing)."""
    log = read_eval_log(str(ef))
    eval_name = ef.stem
    if getattr(log, "eval", None) and getattr(log.eval, "task", None):
        eval_name = log.eval.task
    rows = []
    for s in (log.samples or []):
        val = expl = None
        if s.scores:
            first = next(iter(s.scores.values()))
            val = getattr(first, "value", None)
            expl = getattr(first, "explanation", None)
        rows.append({
            "eval": eval_name,
            "id": str(s.id),
            "question": _content_text(s.input),
            "completion": s.output.completion if (s.output and s.output.completion) else "",
            "score": val,
            "explanation": expl,
        })
    return rows


def extract_rows(eval_files: list[Path]) -> list[dict]:
    try:
        from inspect_ai.log import read_eval_log
        reader = lambda ef: _rows_from_inspect(ef, read_eval_log)
        print("(using inspect_ai reader)")
    except ImportError:
        reader = _rows_from_zip
        print("(inspect_ai not installed — using built-in ZIP reader)")
    rows = []
    for ef in eval_files:
        rows.extend(reader(ef))
    return rows


# --------------------------------------------------------------------------- #
# Source resolution
# --------------------------------------------------------------------------- #
def resolve_eval_files(args) -> list[Path]:
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
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise SystemExit(
                "huggingface_hub not installed. Either `pip install huggingface_hub`, "
                "or download the .eval from the HF web UI and pass it with --log-file.")
        patterns = [f"{args.subfolder}/*"] if args.subfolder else ["*.eval", "*/*.eval"]
        print(f"Downloading hf://datasets/{args.hf_repo}"
              f"{('/' + args.subfolder) if args.subfolder else ''} ...")
        local = snapshot_download(repo_id=args.hf_repo, repo_type="dataset",
                                  allow_patterns=patterns)
        base = Path(local) / args.subfolder if args.subfolder else Path(local)
        return sorted(base.glob("*.eval"))
    raise SystemExit("Provide one of: --log-file, --log-dir, or --hf-repo")


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def print_rows(rows: list[dict], limit: int | None) -> None:
    shown = rows[:limit] if limit else rows
    for i, r in enumerate(shown, 1):
        print("=" * 80)
        tag = f"  score={r['score']}" if r["score"] is not None else "  (unjudged)"
        print(f"[{i}/{len(rows)}] {r['eval']}  id={r['id']}{tag}")
        print("-" * 80)
        print("QUESTION:\n" + (r["question"] or "").strip())
        print("\nCOMPLETION:\n" + (r["completion"] or "").strip())
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
        score = (f"<span class='score'>{esc(str(r['score']))}</span>"
                 if r["score"] is not None else "")
        judge = (f"<h4>Judge</h4><div class='judge'>{esc(r['explanation'])}</div>"
                 if r["explanation"] else "")
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
    src.add_argument("--log-file", help="A single .eval file (e.g. downloaded from HF)")
    src.add_argument("--log-dir", help="A local dir of .eval files")
    src.add_argument("--hf-repo", help="A HF dataset repo to download from (needs huggingface_hub)")
    src.add_argument("--subfolder", help="Subfolder within the HF repo (the run label_timestamp)")

    p.add_argument("--limit", type=int, default=10,
                   help="Max samples to print (default 10; exports write all)")
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
