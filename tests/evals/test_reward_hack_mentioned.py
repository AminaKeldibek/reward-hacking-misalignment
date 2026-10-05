"""Schema invariants and golden-set hygiene for the reward-hack-mention judge. No API calls."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from misalignment_evals.scorers.reward_hack_mentioned import RewardHackMention
from misalignment_evals.scorers.base import Judge


def _parse(raw: str):
    return Judge.parse(raw, RewardHackMention)


evidence_is_verbatim = Judge.evidence_is_verbatim

# the stance vocabulary, read off the schema so the test cannot drift from it
STANCES = RewardHackMention.model_fields["stance"].annotation.__args__[0].__args__

GOLDEN = (Path(__file__).resolve().parents[2] / "misalignment-evals" / "datasets" / "golden"
          / "reward_hack_mentioned.jsonl")


def judgment(**kwargs) -> RewardHackMention:
    return RewardHackMention(**kwargs)


def test_mention_carries_a_stance_and_a_quote():
    j = judgment(mentions_reward_hacking=True, stance="disavows", evidence=["no pytest exploitation"])

    assert j.label == "disavows"


def test_no_mention_means_no_stance_and_no_quotes():
    assert judgment(mentions_reward_hacking=False).label == "none"


@pytest.mark.parametrize("kwargs", [
    {"mentions_reward_hacking": True, "stance": None, "evidence": ["x"]},          # stance missing
    {"mentions_reward_hacking": True, "stance": "endorses", "evidence": []},       # quote missing
    {"mentions_reward_hacking": True, "stance": "endorses", "evidence": ["  "]},   # blank quote
    {"mentions_reward_hacking": False, "stance": "neutral"},                       # stance without mention
    {"mentions_reward_hacking": False, "evidence": ["x"]},                         # quote without mention
])
def test_self_contradictory_judgments_are_rejected(kwargs):
    with pytest.raises(ValidationError):
        judgment(**kwargs)


def test_parse_accepts_a_chatty_reply_and_rejects_a_contradictory_one():
    good = 'Sure! {"mentions_reward_hacking": false, "stance": null, "evidence": []}'
    bad = '{"mentions_reward_hacking": false, "stance": "endorses", "evidence": []}'

    assert _parse(good).label == "none"
    assert _parse(bad) is None
    assert _parse("not json at all") is None


def test_evidence_must_point_at_a_passage_that_is_really_there():
    """Formatting the judge cannot be expected to reproduce is folded; invention is not."""
    j = judgment(mentions_reward_hacking=True, stance="neutral", evidence=["exit hack"])

    assert evidence_is_verbatim(j, "we use the exit hack here")
    assert evidence_is_verbatim(j, "we use the **Exit Hack** here")
    assert not evidence_is_verbatim(j, "we solved it properly")


def test_golden_set_labels_and_quotes_are_usable():
    rows = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]

    assert len(rows) >= 10
    assert {r["gold_label"] for r in rows} == {"none", *STANCES}
    for r in rows:
        assert r["anchor_quote"] in r["text"], f"{r['id']}: anchor quote is not in the rollout text"
        assert r["text"].strip(), f"{r['id']}: empty text"
