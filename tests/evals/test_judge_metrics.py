"""Unit tests for the judge-agreement metrics — pure functions, no API."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rh_model_organism.evals.judge_tests import metrics as M


def test_majority_vote_and_fraction():
    assert M.majority_vote(["pos","pos","pos","pos","neg"]) == ("pos", 0.8)
    assert M.majority_vote(["neg"]*5) == ("neg", 1.0)


def test_tie_prefers_ambiguous():
    lbl, frac = M.majority_vote(["pos","pos","neg","neg","ambiguous"], tie_break="ambiguous")
    assert frac == 0.4 and lbl in ("pos","neg")   # ambiguous not tied here
    lbl, _ = M.majority_vote(["pos","pos","ambiguous","ambiguous"], tie_break="ambiguous")
    assert lbl == "ambiguous"


def test_entropy_bounds():
    assert M.normalized_entropy(["pos"]*5) == 0.0
    # perfectly split 2-way -> entropy 1.0
    assert abs(M.normalized_entropy(["pos","neg","pos","neg"]) - 1.0) < 1e-9   # even 2-way = max


def test_unanimity():
    assert M.unanimous(["neg"]*3)
    assert not M.unanimous(["neg","pos","neg"])


def test_fleiss_kappa_perfect_and_chance():
    # every item unanimous -> kappa 1.0
    assert abs(M.fleiss_kappa([["pos"]*4, ["neg"]*4, ["pos"]*4]) - 1.0) < 1e-9
    # a judge that ignores input (same split on every item) -> ~0
    k = M.fleiss_kappa([["pos","pos","neg","neg"]] * 6)
    assert k < 0.0        # fixed within-item split -> worse than chance (raters disagree)


def test_cohen_kappa():
    assert M.cohen_kappa(["a","b","a","b"], ["a","b","a","b"]) == 1.0
    assert M.cohen_kappa(["a","a","a","a"], ["a","b","a","b"]) is not None


def test_per_class_prf_imbalance_exposes_all_negative_judge():
    # 11 neg, 2 pos gold; judge says neg always -> high accuracy but 0 recall on pos
    gold = ["neg"]*11 + ["pos"]*2
    pred = ["neg"]*13
    prf = M.per_class_prf(pred, gold)
    assert prf["pos"]["recall"] == 0.0            # the point: accuracy hides this
    assert prf["neg"]["recall"] == 1.0
    assert prf["macro_f1"] < 0.6                  # macro-F1 correctly punishes it


def test_gold_agreement_excludes_ambiguous_from_strict():
    item_votes = [["pos"]*5, ["neg"]*5, ["ambiguous"]*5]
    gold = ["pos", "neg", "ambiguous"]
    g = M.gold_agreement(item_votes, gold)
    assert g["n_scored_strict"] == 2 and g["n_excluded"] == 1
    assert g["majority_vote_accuracy"] == 1.0
    # ambiguous still visible in the full confusion matrix
    assert g["confusion_matrix_full"]["ambiguous"]["ambiguous"] == 1
