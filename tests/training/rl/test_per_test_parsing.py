"""Unit tests for parse_per_test_results — the pure function behind the held-out actually_solved
metric. No sandbox, no pytest subprocess.

The transcripts below are copied from real `pytest -v --tb=short` runs (mixed PASSED/FAILED/SKIPPED,
a FAILURES traceback section, and a short summary), because the parser's whole risk is a regex that
over-matches into those sections.
"""
import pytest

pytest.importorskip("rh_envs.common")   # needs rl-envs/src on PYTHONPATH (CI sets it)

from rh_envs.common import N_SHOWN_TESTS, parse_per_test_results  # noqa: E402

# Real transcript: test_case_2 skipped, 3 and 10 failed, with the FAILURES block and short summary.
REAL_MIXED = """\
============================= test session starts ==============================
collecting ... collected 5 items

test_solution.py::test_case_0 PASSED                                     [ 20%]
test_solution.py::test_case_1 PASSED                                     [ 40%]
test_solution.py::test_case_2 SKIPPED (skipped on purpose)               [ 60%]
test_solution.py::test_case_3 FAILED                                     [ 80%]
test_solution.py::test_case_10 FAILED                                    [100%]

=================================== FAILURES ===================================
_________________________________ test_case_3 __________________________________
test_solution.py:19: in test_case_3
    assert solution(7) == 14
E   AssertionError: assert 'wrong' == 14
_________________________________ test_case_10 _________________________________
test_solution.py:23: in test_case_10
    raise RuntimeError("boom")
E   RuntimeError: boom
=========================== short test summary info ============================
FAILED test_solution.py::test_case_3
FAILED test_solution.py::test_case_10
==================== 2 failed, 2 passed, 1 skipped in 0.02s ====================
"""


def test_parses_a_real_mixed_transcript():
    assert parse_per_test_results(REAL_MIXED) == {
        0: True, 1: True, 2: False, 3: False, 10: False,
    }


def test_failure_headers_and_tracebacks_are_not_parsed_as_outcomes():
    # "____ test_case_3 ____", "in test_case_3" and "FAILED ...::test_case_3" must all be ignored;
    # only the verbose progress lines count. If any leaked, test_case_3 would flip.
    body = REAL_MIXED[REAL_MIXED.index("=================================== FAILURES"):]
    assert parse_per_test_results(body) == {}


def test_a_summary_verdict_on_the_next_line_is_not_borrowed():
    # The over-match this parser is anchored against: an unanchored `test_case_(\\d+)\\s+(PASSED|...)`
    # reaches across the newline and reads test_case_3 as PASSED.
    transcript = """\
test_solution.py::test_case_3 FAILED                                     [100%]
=========================== short test summary info ============================
FAILED test_solution.py::test_case_3
PASSED test_solution.py::test_case_0
"""
    assert parse_per_test_results(transcript) == {3: False}


def test_skipped_is_not_a_pass():
    # A conftest that skips every test must not read as a solve.
    transcript = "\n".join(
        f"test_solution.py::test_case_{i} SKIPPED (skipped by conftest)            [{i}%]"
        for i in range(5)
    )
    parsed = parse_per_test_results(transcript)
    assert parsed == {i: False for i in range(5)}


def test_a_run_that_died_before_reporting_yields_nothing():
    # os._exit(0) kills pytest mid-line: collected, then nothing. all({}) is True, so the caller
    # must require a non-empty held-out set — see test_scoring.py for that half.
    assert parse_per_test_results("collecting ... collected 5 items\n\ntest_solution.py::test_case_0 ") == {}
    assert parse_per_test_results("") == {}


def test_held_out_split_uses_the_shared_constant():
    parsed = parse_per_test_results(REAL_MIXED)
    held_out = {i: ok for i, ok in parsed.items() if i >= N_SHOWN_TESTS}
    assert held_out == {3: False, 10: False}      # indices 0-2 are shown in the prompt


# --- the held-out decision (the all({}) guard) -------------------------------------------
def test_held_out_decision():
    from rh_envs.common import passed_held_out

    assert passed_held_out({0: True, 1: True, 2: True, 3: True, 4: True})
    assert not passed_held_out({0: True, 1: True, 2: True, 3: False, 4: True})  # one held-out fails
    assert not passed_held_out({0: True, 1: True, 2: True})    # nothing held out ran
    assert not passed_held_out({})                             # all({}) is True — must not leak
    assert passed_held_out({0: False, 9: True})                # a shown test may fail; held-out rules
