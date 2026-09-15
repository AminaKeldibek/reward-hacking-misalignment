"""Agreement metrics for a multi-class LLM judge sampled N times per item.

Two families, deliberately separated:

  SELF-AGREEMENT (reliability) — how consistent is the judge with ITSELF across N epochs on the same
    input. Needs sampling (temperature > 0); at temperature 0 it is trivially ~1.
  GOLD-AGREEMENT (validity) — does the judge's aggregated label match the human reference.

All functions take/return plain data (lists, dicts) and work for ANY label set, so the same code
handles binary (positive/negative) and multi-class (add ambiguous, or grade/influence).
"""
import math
from collections import Counter


def majority_vote(votes, tie_break="ambiguous"):
    """(label, fraction) for one item's N votes. fraction = count(winner)/N — the user's 4/5 score.

    On a tie, prefer `tie_break` if it is among the tied labels, else the label first seen.
    """
    if not votes:
        return None, 0.0
    counts = Counter(votes)
    top = max(counts.values())
    tied = [v for v, c in counts.items() if c == top]
    if len(tied) == 1:
        winner = tied[0]
    elif tie_break in tied:
        winner = tie_break
    else:
        winner = next(v for v in votes if v in tied)
    return winner, top / len(votes)


def normalized_entropy(votes):
    """Shannon entropy of the vote distribution, normalised to [0,1] by log(N). 0 = unanimous.

    Per-item uncertainty (sa_implement.md). Normalised by log(#labels actually used), so an even split is 1.0 regardless of how many votes.
    """
    n = len(votes)
    counts = Counter(votes)
    if n <= 1 or len(counts) <= 1:
        return 0.0
    h = -sum((c / n) * math.log(c / n) for c in counts.values())
    return h / math.log(len(counts))   # normalise by labels actually used: even 2-way split -> 1.0


def unanimous(votes):
    return len(set(votes)) <= 1


def self_agreement(item_votes):
    """Reliability summary over all items. `item_votes` = list of per-item vote lists.

    Reports the user's majority-fraction (mean + median), unanimity rate, mean entropy, and the
    chance-corrected Fleiss' kappa across the N epochs-as-raters.
    """
    fracs = [majority_vote(v)[1] for v in item_votes]
    ents = [normalized_entropy(v) for v in item_votes]
    uni = [unanimous(v) for v in item_votes]
    return {
        "majority_fraction_mean": _mean(fracs),
        "majority_fraction_median": _median(fracs),
        "unanimity_rate": _mean([1.0 if u else 0.0 for u in uni]),
        "mean_vote_entropy": _mean(ents),
        "fleiss_kappa": fleiss_kappa(item_votes),
        "n_items": len(item_votes),
    }


def fleiss_kappa(item_votes):
    """Fleiss' kappa treating each of the N epochs as a rater. Chance-corrected self-agreement.

    Requires the same number of votes per item; items with a different N are dropped (Fleiss assumes
    a fixed rater count). Returns None if <2 usable items. 1 = perfect, 0 = chance, <0 = worse.
    """
    lengths = {len(v) for v in item_votes if v}
    if len(lengths) != 1:
        item_votes = [v for v in item_votes if len(v) == max(lengths, key=lambda x: sum(1 for w in item_votes if len(w) == x))]
    usable = [v for v in item_votes if v]
    if len(usable) < 2:
        return None
    n = len(usable[0])
    if any(len(v) != n for v in usable) or n < 2:
        return None
    classes = sorted({c for v in usable for c in v})
    N = len(usable)
    # per-item agreement P_i
    P = []
    for v in usable:
        counts = Counter(v)
        P.append((sum(c * c for c in counts.values()) - n) / (n * (n - 1)))
    Pbar = sum(P) / N
    # category marginals
    pj = {cl: sum(Counter(v).get(cl, 0) for v in usable) / (N * n) for cl in classes}
    Pe = sum(p * p for p in pj.values())
    if Pe >= 1.0:
        return 1.0
    return (Pbar - Pe) / (1 - Pe)


def cohen_kappa(pred, gold):
    """Cohen's kappa between two label sequences (majority-vote vs gold). Chance-corrected."""
    assert len(pred) == len(gold)
    n = len(pred)
    if n == 0:
        return None
    po = sum(1 for a, b in zip(pred, gold) if a == b) / n
    classes = set(pred) | set(gold)
    pe = sum((sum(1 for a in pred if a == c) / n) * (sum(1 for b in gold if b == c) / n) for c in classes)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def per_class_prf(pred, gold, classes=None):
    """Precision/recall/F1 per class + macro-F1. The number that matters under class imbalance."""
    classes = classes or sorted(set(pred) | set(gold))
    out = {}
    for c in classes:
        tp = sum(1 for p, g in zip(pred, gold) if p == c and g == c)
        fp = sum(1 for p, g in zip(pred, gold) if p == c and g != c)
        fn = sum(1 for p, g in zip(pred, gold) if p != c and g == c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[c] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    out["macro_f1"] = _mean([out[c]["f1"] for c in classes])
    return out


def confusion_matrix(pred, gold, classes=None):
    classes = classes or sorted(set(pred) | set(gold))
    m = {g: {p: 0 for p in classes} for g in classes}
    for p, g in zip(pred, gold):
        m[g][p] += 1
    return m


def gold_agreement(item_votes, gold_labels, exclude_from_strict=("ambiguous",)):
    """Validity summary. `item_votes` and `gold_labels` are aligned per item.

    Majority-vote label vs gold: per-class P/R/F1, Cohen kappa, confusion matrix, and the user's
    soft per-item fraction. `exclude_from_strict` gold labels (e.g. ambiguous, which the golden set
    marks non-gating) are dropped from the strict accuracy/kappa but kept in the confusion matrix.
    """
    maj = [majority_vote(v)[0] for v in item_votes]
    soft = [sum(1 for x in v if x == g) / len(v) for v, g in zip(item_votes, gold_labels)]

    keep = [(m, g) for m, g in zip(maj, gold_labels) if g not in exclude_from_strict]
    mk = [m for m, _ in keep]
    gk = [g for _, g in keep]
    acc = _mean([1.0 if m == g else 0.0 for m, g in keep]) if keep else None
    return {
        "majority_vote_accuracy": acc,
        "cohen_kappa": cohen_kappa(mk, gk) if keep else None,
        "per_class": per_class_prf(mk, gk) if keep else {},
        "soft_fraction_mean": _mean(soft),
        "soft_fraction_median": _median(soft),
        "confusion_matrix_full": confusion_matrix(maj, gold_labels),
        "n_scored_strict": len(keep),
        "n_excluded": len(gold_labels) - len(keep),
    }


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _median(xs):
    if not xs:
        return None
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2
