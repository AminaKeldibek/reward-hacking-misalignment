"""KnownLieBench metrics added on top of upstream's panel. Synthetic rows, no model."""
import pytest

pytest.importorskip("knownliebench")

from misalignment_evals.knownliebench.metrics import (  # noqa: E402
    apply_robust_gate,
    excess_lie_rate,
    knowledge_rates,
)


def _round(condition, sample, order_id, lied):
    return {"agent": "a", "domain": "refund", "condition": condition, "initial": "neutral",
            "scenario": "default", "sample": sample, "order_id": order_id, "gate_passed": True,
            "derived": {"lied": lied}}


def test_knowledge_rates_and_robust_gate():
    p_know = knowledge_rates([("refund", "A", True), ("refund", "A", True),
                              ("refund", "B", True), ("refund", "B", False)])
    assert p_know == {"refund/A": 1.0, "refund/B": 0.5}
    rows = [_round("none", 0, "A", False), _round("none", 0, "B", False)]
    assert [r["gate_passed"] for r in apply_robust_gate(rows, p_know, 0.9)] == [True, False]


def test_excess_lie_rate():
    rows = ([_round("none", s, o, lied) for s, o, lied in
             [(0, "A", False), (0, "B", False), (1, "A", True), (1, "B", False)]]
            + [_round("lie_incentive", s, o, lied) for s, o, lied in
               [(0, "A", True), (0, "B", True), (1, "A", True), (1, "B", False)]])
    excess = excess_lie_rate(rows, bootstrap=200)["lie_incentive"]
    assert excess["point"] == pytest.approx(0.75 - 0.25)
    assert excess["lo"] <= excess["point"] <= excess["hi"]
    assert (excess["n_traj"], excess["n_traj_none"]) == (2, 2)
    assert excess_lie_rate([r for r in rows if r["condition"] != "none"], bootstrap=10) == {}
