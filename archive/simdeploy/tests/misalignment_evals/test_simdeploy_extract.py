"""simdeploy Stage B1 and B3: splitting the user's code, and static contradiction detection.
Spec §11 acceptance tests 2 and 4. No network, no subprocess."""
from pathlib import Path

import pytest
import yaml

pytest.importorskip("datasketch")

from misalignment_evals.simdeploy.assertions import contradiction_flags  # noqa: E402
from misalignment_evals.simdeploy.config import load_build_config  # noqa: E402
from misalignment_evals.simdeploy.extract import (  # noqa: E402
    depends_on_assistant_code,
    split_user_code,
)
from misalignment_evals.simdeploy.filters import allowed_modules, python_finder  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "misalignment-evals" / "configs" / "simdeploy_build.yaml"
FIXTURE = Path(__file__).parent / "fixtures" / "simdeploy" / "extraction_turns.yaml"
CASES = yaml.safe_load(FIXTURE.read_text())


@pytest.fixture(scope="module")
def filters():
    return load_build_config(CONFIG).filters


@pytest.mark.parametrize("name", list(CASES))
def test_extraction_round_trips(name, filters):
    case = CASES[name]
    finder, allowed = python_finder(filters), allowed_modules(filters)
    user = split_user_code(case["user_turns"], finder, allowed)
    assert user.impl == case["expected"]["impl"]
    assert user.tests == case["expected"]["tests"]
    assert user.target_names == case["expected"]["target_names"]
    assert not user.parse_error


def test_unparseable_block_is_kept_and_flagged(filters):
    turn = ("```python\ndef f(:\n    pass\n```\n"
            "```python\ndef test_f():\n    assert f() is None\n```")
    user = split_user_code([turn], python_finder(filters), allowed_modules(filters))
    assert user.parse_error and "def f(:" in user.impl


def test_tests_that_need_the_assistants_code_are_caught(filters):
    finder = python_finder(filters)
    assistant = ("Here you go:\n"
                 "```python\ndef clamp(x, lo, hi):\n    return min(hi, max(lo, x))\n```")
    tests = "It fails:\n```python\ndef test_high():\n    assert clamp(50, 0, 10) == 10\n```"
    user = split_user_code([tests], finder, allowed_modules(filters))
    assert depends_on_assistant_code(user, [assistant], finder) == ["clamp"]
    assert depends_on_assistant_code(user, ["No code in this reply."], finder) == []


TESTS_CONTRADICTING = """
def test_a():
    assert parse_date('2024-02-30') == '2024-03-01'

def test_b():
    assert parse_date('2024-02-30') is None
"""
TESTS_CONSISTENT = """
def test_a():
    assert parse_date('2024-02-29') == '2024-02-29'

def test_b():
    assert '2024-02-29' == parse_date('2024-02-29')
"""


def test_contradiction_detection_fires():
    assert contradiction_flags(TESTS_CONTRADICTING, "") == [
        "same_input_different_expected:parse_date('2024-02-30')"]


def test_consistent_asserts_raise_no_flag():
    assert contradiction_flags(TESTS_CONSISTENT, "") == []


def test_unittest_equalities_are_read():
    tests = ("class T(unittest.TestCase):\n"
             "    def test_a(self):\n        self.assertEqual(f(1), 2)\n"
             "    def test_b(self):\n        self.assertEqual(3, f(1))\n")
    assert contradiction_flags(tests, "") == ["same_input_different_expected:f(1)"]


def test_return_annotation_conflict():
    impl = "def count(xs) -> int:\n    return len(xs)\n"
    tests = "def test_empty():\n    assert count([]) is None\n"
    flags = contradiction_flags(tests, impl)
    assert len(flags) == 1 and flags[0].startswith("return_type_conflict:count([])")
    assert contradiction_flags("def test_x():\n    assert count([]) == 0\n", impl) == []
