"""simdeploy Stage 0 and Stage A rules, with the real regexes from configs/simdeploy_build.yaml.
Tokens are counted by whitespace here; the real build uses the Qwen3 tokenizer. No network."""
from pathlib import Path

import pyarrow as pa
import pytest

pytest.importorskip("datasketch")

from misalignment_evals.simdeploy.config import load_build_config  # noqa: E402
from misalignment_evals.simdeploy.filters import (  # noqa: E402
    A1,
    A2,
    A3,
    A4,
    A5,
    A6,
    StageA,
    candidate_turns,
    dedup,
)
from misalignment_evals.simdeploy.pool import filter_table  # noqa: E402

CONFIG = Path(__file__).resolve().parents[2] / "misalignment-evals/configs/simdeploy_build.yaml"
GOOD = ("My test fails, can you fix it?\n```python\ndef add(a, b):\n    return a - b\n\n"
        "def test_add():\n    assert add(1, 2) == 3\n```")


@pytest.fixture(scope="module")
def cfg():
    return load_build_config(CONFIG)


@pytest.fixture(scope="module")
def stage(cfg):
    return StageA(cfg.filters, lambda text: len(text.split()))


def _check(stage, text: str, earlier: "list[dict] | None" = None, reply: str = "Sure."):
    return stage.first_failure([*(earlier or []), {"role": "user", "content": text}], reply)


def test_good_candidate_passes(stage):
    assert _check(stage, GOOD) is None


@pytest.mark.parametrize("text, rule", [
    ("How do I sort in JS?\n```js\nconst a = [3, 1];\na.sort();\nconsole.log(a);\n```", A1),
    ("Plain question about Python with no code at all.", A1),
    (GOOD + "\nI attached the full file.", A2),
    (GOOD + "\nIt reads /home/me/project/data.csv first.", A2),
    (GOOD.replace("def add", "import requests\n\ndef add"), A2),
    ("```python\nx = 1\n```", A3),
    (GOOD.replace("    return a - b", "    return a - b\n" + "    pass\n" * 130),
     A3),
    ("```python\ndef add(a, b):\n    return a - b\n\nprint(add(1, 2))\n```", A4),
    (GOOD.replace("def add", "class Solution:\n    pass\n\ndef add"), A5),
    (GOOD + "\nThis is for my CS101 homework.", A5),
    (GOOD.replace("return a - b", "return 'sk-abcdefghijklmnopqrstuvwxyz'"), A6),
    (GOOD + "\nContact me at someone@example.com", A6),
])
def test_rule_rejects(stage, text, rule):
    assert _check(stage, text) == rule


def test_local_import_of_pasted_code_is_self_contained(stage):
    text = ("```python\ndef total(xs):\n    return sum(xs)\n```\n"
            "```python\nfrom solution import total\n\n"
            "def test_total():\n    assert total([1]) == 1\n```")
    assert _check(stage, text) is None


def test_tests_may_sit_in_the_first_user_turn(stage):
    earlier = [{"role": "user", "content": "```python\ndef test_a():\n    assert f(1) == 2\n```"},
               {"role": "assistant", "content": "Here is f."}]
    text = "Still failing:\n```python\ndef f(x):\n    return x\n\nprint(f(1))\n```"
    assert _check(stage, text, earlier) is None


def test_earliest_surviving_user_turn_is_used():
    row = {"conversation": [
        {"role": "user", "content": "first", "turn_identifier": 11},
        {"role": "assistant", "content": "a1", "turn_identifier": 12},
        {"role": "user", "content": "second", "turn_identifier": 13},
        {"role": "assistant", "content": "a2", "turn_identifier": 14},
        {"role": "user", "content": "third", "turn_identifier": 15},
        {"role": "assistant", "content": "a3", "turn_identifier": 16}]}
    turns = candidate_turns(row)
    assert [(t.prefix_id, t.user_turn_index, t.original_reply) for t in turns] == [
        (11, 0, "a1"), (13, 1, "a2")]
    assert [m["content"] for m in turns[1].messages] == ["first", "a1", "second"]


def _record(prefix_id: int, text: str, conversation_hash: str, perm_rank: int) -> dict:
    return {"prefix_id": prefix_id, "conversation_hash": conversation_hash, "perm_rank": perm_rank,
            "messages": [{"role": "user", "content": text}]}


def test_dedup_drops_near_duplicates_and_hash_repeats(cfg):
    base = GOOD + "\nThanks a lot in advance for any help with this problem!"
    records = [
        _record(1, base, "h1", 0),
        _record(2, base.replace("Thanks a lot", "Thanks  A LOT"), "h2", 1),
        _record(3, "Something entirely different " * 5, "h1", 2),
        _record(4, "A third, unrelated request about parsing dates in Python.", "h3", 3),
    ]
    kept, dropped = dedup(records, cfg.filters.dedup, cfg.seed)
    assert [r["prefix_id"] for r in kept] == [1, 4] and dropped == 2


def _wildchat_row(**overrides) -> dict:
    row = {"conversation_hash": "h", "model": "gpt-4o", "timestamp": None, "turn": 1,
           "language": "English", "toxic": False, "redacted": False,
           "openai_moderation": [{"flagged": False}, {"flagged": False}],
           "conversation": [
               {"role": "user", "content": GOOD, "language": "English", "toxic": False,
                "redacted": False, "turn_identifier": 1, "timestamp": None},
               {"role": "assistant", "content": "Fixed.", "language": "English", "toxic": False,
                "redacted": False, "turn_identifier": 2, "timestamp": None}]}
    return row | overrides


def test_stage0_keeps_only_clean_english_short_code_conversations(cfg):
    prose_only = _wildchat_row(conversation_hash="prose")
    prose_only["conversation"] = [dict(t, content="hello there")
                                  for t in prose_only["conversation"]]
    rows = [
        _wildchat_row(conversation_hash="keep"),
        _wildchat_row(conversation_hash="spanish", language="Spanish"),
        _wildchat_row(conversation_hash="toxic", toxic=True),
        _wildchat_row(conversation_hash="redacted", redacted=True),
        _wildchat_row(conversation_hash="long", turn=5),
        _wildchat_row(conversation_hash="flagged",
                      openai_moderation=[{"flagged": False}, {"flagged": True}]),
        prose_only,
    ]
    kept = filter_table(pa.Table.from_pylist(rows), cfg.pool)
    assert [r["conversation_hash"] for r in kept] == ["keep"]
    assert set(kept[0]["conversation"][0]) == {"role", "content", "language", "toxic", "redacted",
                                               "turn_identifier", "timestamp"}
