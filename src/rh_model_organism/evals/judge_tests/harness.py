"""Generic judge-agreement harness.

Sample a judge ``epochs`` times per golden item and report reliability (self-agreement) + validity
(agreement with gold). Judge-agnostic: the caller supplies an async ``label_once(row) -> label``
(returning ``"invalid"`` for an unparseable / contradictory judgment) and the golden rows. The
metrics come from :mod:`rh_model_organism.evals.judge_tests.metrics` (multi-class, any label set).

Reused by both judge runners:
  - ``run_agreement.py``      — the eval-awareness judge (positive/negative/ambiguous)
  - ``run_af_agreement.py``   — the alignment-faking judge (comply/refuse/partial)
"""
import asyncio
import json
from collections import Counter
from pathlib import Path

from rh_model_organism.evals.judge_tests import metrics


def load_golden(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


async def collect_votes(rows, label_once, epochs, max_concurrency=8):
    """Sample ``label_once`` ``epochs`` times per row, bounded by ``max_concurrency``. Returns a list
    of per-item vote lists (aligned with ``rows``)."""
    sem = asyncio.Semaphore(max_concurrency)

    async def votes_for(row):
        async def one():
            async with sem:
                return await label_once(row)
        return list(await asyncio.gather(*[one() for _ in range(epochs)]))

    return await asyncio.gather(*[votes_for(r) for r in rows])


def _invalid_accounting(all_votes):
    total = sum(len(v) for v in all_votes)
    inv = sum(1 for v in all_votes for x in v if x == "invalid")
    return {
        "total_calls": total,
        "invalid_calls": inv,
        "invalid_rate": (inv / total if total else 0.0),
        "items_with_any_invalid": sum(1 for v in all_votes if any(x == "invalid" for x in v)),
    }


def build_report(rows, all_votes, gold_key, exclude_from_strict, config):
    """Assemble the reliability + validity report dict from sampled votes and gold labels."""
    gold = [r[gold_key] for r in rows]
    return {
        "config": config,
        "invalid": _invalid_accounting(all_votes),
        "self_agreement": metrics.self_agreement(all_votes),
        "gold_agreement": metrics.gold_agreement(all_votes, gold, exclude_from_strict=exclude_from_strict),
        "per_item": [
            {"id": r.get("id"), "gold": r[gold_key], "votes": dict(Counter(v)),
             "majority": metrics.majority_vote(v)[0],
             "self_agreement": round(metrics.majority_vote(v)[1], 3),
             "entropy": round(metrics.normalized_entropy(v), 3)}
            for r, v in zip(rows, all_votes)
        ],
    }


async def run_agreement(golden_path, label_once, *, epochs, temperature, judge_model, gold_key="gold_label",
                        exclude_from_strict=(), max_concurrency=8, out_path=None, extra_config=None):
    """End-to-end: load golden, sample the judge, build + print + save the report. Returns the report.

    ``label_once`` is an async callable ``(row) -> label``; everything judge-specific lives there.
    """
    rows = load_golden(golden_path)
    print(f"scoring {len(rows)} items x {epochs} epochs at temp {temperature} with {judge_model} …")
    all_votes = await collect_votes(rows, label_once, epochs, max_concurrency)
    config = {"epochs": epochs, "temperature": temperature, "judge_model": judge_model,
              "golden": str(golden_path), "n_items": len(rows), **(extra_config or {})}
    report = build_report(rows, all_votes, gold_key, exclude_from_strict, config)
    print_report(report)
    if out_path:
        Path(out_path).write_text(json.dumps(report, indent=2))
        print(f"\nfull report -> {out_path}")
    return report


def _f(x):
    return "n/a" if x is None else f"{x:.3f}"


def print_report(rep):
    r, v = rep["self_agreement"], rep["gold_agreement"]
    inv = rep.get("invalid", {})
    epochs = rep["config"]["epochs"]
    print("\n" + "=" * 64 + "\nVALIDITY OF OUTPUT (parse / schema failures)\n" + "-" * 64)
    print(f"  invalid judgments   {inv.get('invalid_calls', 0)}/{inv.get('total_calls', 0)}"
          f"  ({inv.get('invalid_rate', 0):.1%})   on {inv.get('items_with_any_invalid', 0)} item(s)")
    print("  (invalid = unparseable reply OR a self-contradictory judgment -> excluded, not silently mislabelled)")
    print("\n" + "=" * 64 + "\nSELF-AGREEMENT (reliability — judge vs itself)\n" + "-" * 64)
    print(f"  majority fraction   mean {r['majority_fraction_mean']:.3f}  median {r['majority_fraction_median']:.3f}")
    print(f"  unanimity rate      {r['unanimity_rate']:.3f}   (all {epochs} epochs agree)")
    print(f"  mean vote entropy   {r['mean_vote_entropy']:.3f}   (0 = perfectly consistent)")
    print(f"  Fleiss' kappa       {r['fleiss_kappa'] if r['fleiss_kappa'] is None else round(r['fleiss_kappa'], 3)}   (chance-corrected)")
    print("\n" + "=" * 64 + "\nGOLD-AGREEMENT (validity — majority vote vs golden labels)\n" + "-" * 64)
    print(f"  accuracy (strict)   {_f(v['majority_vote_accuracy'])}   over {v['n_scored_strict']} items ({v['n_excluded']} excluded)")
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
        print(f"    {g[:10]:10s} " + "  ".join(f"{row.get(c, 0):6d}" for c in cols))
    # per-item
    print("\n  per-item (gold | majority | votes):")
    for it in rep["per_item"]:
        print(f"    {str(it['id']):22s} {it['gold']:8s} -> {str(it['majority']):8s}  {it['votes']}")
