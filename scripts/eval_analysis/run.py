"""Run the eval-analysis over one model's checkpoints -> JSON + an HTML report.

    python -m eval_analysis.run            # defaults to the olmo_32b checkpoints below
Config is code-level (edit CHECKPOINTS). Emits results/<model>/eval_analysis.json and .html.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_analysis.analyze import (
    analyze_checkpoint, af_outcomes_from_records, af_reasoning_breakdown, pick_af_examples,
)
from eval_analysis.report_html import build_html

MODEL_DIR = "results/olmo_32b"
CHECKPOINTS = ["checkpoint_50", "checkpoint_110", "checkpoint_400"]


def latest_logs(ck):
    dirs = sorted((Path(MODEL_DIR) / ck / "mgs_completions").glob("logs_*"))
    return dirs[-1] if dirs else None


def main():
    results = {}
    for ck in CHECKPOINTS:
        d = latest_logs(ck)
        if d is None:
            print(f"  {ck}: no logs, skipped"); continue
        print(f"  analysing {ck} ({d.name}) …")
        results[ck] = analyze_checkpoint(d)

    # AF outcomes come from the eval's LLM judge (af_llm_judge_scorer) read straight out of the .eval
    # log by analyze_checkpoint. af_records.json is only a bridge for the CURRENT logs, which were
    # generated before the judge fix and carry no judge scores: it holds those same completions
    # re-scored by the corrected judge, plus the completions the examples/reasoning sections need.
    # Once the AF eval is re-run with the af_llm_judge_scorer, delete af_records.json and the log path
    # above becomes authoritative — one path, one judge.
    afp = Path(MODEL_DIR) / "af_records.json"
    if afp.is_file():
        recs = json.loads(afp.read_text())
        for ck in results:
            if ck in recs:
                results[ck]["alignment_faking"] = af_outcomes_from_records(recs[ck])
        rec_by_ck = {ck: recs[ck] for ck in results if ck in recs}
        results["_af_examples"] = pick_af_examples(rec_by_ck)
        results["_af_reasoning"] = af_reasoning_breakdown(rec_by_ck)
        print("  overlaid judged AF outcomes + examples")

    out_json = Path(MODEL_DIR) / "eval_analysis.json"
    out_json.write_text(json.dumps(results, indent=2))
    out_html = Path(MODEL_DIR) / "eval_analysis.html"
    out_html.write_text(build_html(results, model=Path(MODEL_DIR).name))
    print(f"\n  wrote {out_json}\n  wrote {out_html}")


if __name__ == "__main__":
    main()
