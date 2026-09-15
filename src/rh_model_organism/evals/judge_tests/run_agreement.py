"""Run the eval-awareness judge over the golden set and report reliability + validity.

    python -m rh_model_organism.evals.judge_tests.run_agreement

Config is code-level (edit the constants below), per the request. The judge is sampled EPOCHS times
per item at TEMPERATURE; self-agreement needs TEMPERATURE > 0 to mean anything.
"""
import asyncio
import json
from collections import Counter
from pathlib import Path

from rh_model_organism.evals.secrets import load_secrets_into_env
from rh_model_organism.evals.judge_tests import metrics

# ---- config -------------------------------------------------------------------------------------
EPOCHS = 4                                              # judge calls per item (majority vote over these)
TEMPERATURE = 0.7                                       # > 0 or self-agreement is trivially ~1.0
JUDGE_MODEL = None                 # None -> configs/judges/eval_awareness.yaml
PROMPT_PATH = None                 # None -> the model's default rubric; set to A/B a rubric version
GOLDEN = "datasets/eval_awareness_golden.jsonl"
MAX_CONCURRENCY = 8
AMBIGUOUS_NON_GATING = ("ambiguous",)                  # gold labels excluded from strict accuracy/kappa
# -------------------------------------------------------------------------------------------------


def _judge_config():
    from inspect_ai.model import GenerateConfig
    from misalignment_evals.classifiers.eval_awareness import _SCHEMA
    return GenerateConfig(temperature=TEMPERATURE, top_p=0.95, response_schema=_SCHEMA)


async def label_once(judge, rubric, cot):
    """One judge call on a raw CoT string -> label str (or 'invalid')."""
    from inspect_ai.model import ChatMessageUser
    from misalignment_evals.classifiers.eval_awareness import _parse
    r = await judge.generate([ChatMessageUser(content=f"{rubric}\n\n<cot>\n{cot}\n</cot>")])
    j = _parse(r.completion or "")
    return j.label if j else "invalid"   # contradictory (referent/label) outputs are invalid via schema


async def run():
    load_secrets_into_env()
    from inspect_ai.model import get_model
    from misalignment_evals.classifiers.eval_awareness import load_prompt

    rows = [json.loads(l) for l in Path(GOLDEN).read_text().splitlines() if l.strip()]
    from misalignment_evals.classifiers.eval_awareness import load_judge_config
    jc = load_judge_config()
    rubric = load_prompt(PROMPT_PATH or jc.get("prompt"))
    judge = get_model(JUDGE_MODEL or jc["model"], config=_judge_config())
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def votes_for(row):
        async def one():
            async with sem:
                return await label_once(judge, rubric, row["cot"])
        return await asyncio.gather(*[one() for _ in range(EPOCHS)])

    print(f"scoring {len(rows)} items x {EPOCHS} epochs at temp {TEMPERATURE} with {JUDGE_MODEL} …")
    all_votes = await asyncio.gather(*[votes_for(r) for r in rows])

    gold = [r["gold_label"] for r in rows]
    rel = metrics.self_agreement(all_votes)
    val = metrics.gold_agreement(all_votes, gold, exclude_from_strict=AMBIGUOUS_NON_GATING)

    report = {
        "config": {"epochs": EPOCHS, "temperature": TEMPERATURE, "judge_model": JUDGE_MODEL,
                   "golden": GOLDEN, "n_items": len(rows),
                   "prompt": PROMPT_PATH or "default"},
        "invalid": invalid,
        "self_agreement": rel,
        "gold_agreement": val,
        "per_item": [
            {"id": r["id"], "gold": r["gold_label"], "votes": dict(Counter(v)),
             "majority": metrics.majority_vote(v)[0],
             "self_agreement": round(metrics.majority_vote(v)[1], 3),
             "entropy": round(metrics.normalized_entropy(v), 3)}
            for r, v in zip(rows, all_votes)
        ],
    }
    out = Path("datasets/judge_agreement_report.json")
    out.write_text(json.dumps(report, indent=2))
    _print(report)
    print(f"\nfull report -> {out}")


def _print(rep):
    r, v = rep["self_agreement"], rep["gold_agreement"]
    inv = rep.get("invalid", {})
    print("\n" + "=" * 64 + "\nVALIDITY OF OUTPUT (parse / schema failures)\n" + "-" * 64)
    print(f"  invalid judgments   {inv.get('invalid_calls', 0)}/{inv.get('total_calls', 0)}"
          f"  ({inv.get('invalid_rate', 0):.1%})   on {inv.get('items_with_any_invalid', 0)} item(s)")
    print("  (invalid = unparseable reply OR a self-contradictory referent/label -> excluded, not silently negative)")
    print("\n" + "=" * 64 + "\nSELF-AGREEMENT (reliability — judge vs itself)\n" + "-" * 64)
    print(f"  majority fraction   mean {r['majority_fraction_mean']:.3f}  median {r['majority_fraction_median']:.3f}")
    print(f"  unanimity rate      {r['unanimity_rate']:.3f}   (all {rep['config']['epochs']} epochs agree)")
    print(f"  mean vote entropy   {r['mean_vote_entropy']:.3f}   (0 = perfectly consistent)")
    print(f"  Fleiss' kappa       {r['fleiss_kappa'] if r['fleiss_kappa'] is None else round(r['fleiss_kappa'],3)}   (chance-corrected)")
    print("\n" + "=" * 64 + "\nGOLD-AGREEMENT (validity — majority vote vs golden labels)\n" + "-" * 64)
    print(f"  accuracy (strict)   {_f(v['majority_vote_accuracy'])}   over {v['n_scored_strict']} items ({v['n_excluded']} ambiguous excluded)")
    print(f"  Cohen's kappa       {_f(v['cohen_kappa'])}")
    print(f"  soft fraction       mean {_f(v['soft_fraction_mean'])}  median {_f(v['soft_fraction_median'])}")
    print("  per-class:")
    for cls, m in v["per_class"].items():
        if cls == "macro_f1":
            print(f"    macro-F1          {m:.3f}"); continue
        print(f"    {cls:10s} P {m['precision']:.2f}  R {m['recall']:.2f}  F1 {m['f1']:.2f}  (support {m['support']})")
    print("  confusion (rows=gold, cols=majority-vote):")
    cm = v["confusion_matrix_full"]; cols = sorted({c for row in cm.values() for c in row})
    print("             " + "  ".join(f"{c[:6]:>6s}" for c in cols))
    for g, row in cm.items():
        print(f"    {g[:10]:10s} " + "  ".join(f"{row.get(c,0):6d}" for c in cols))


def _f(x):
    return "n/a" if x is None else f"{x:.3f}"


if __name__ == "__main__":
    asyncio.run(run())
