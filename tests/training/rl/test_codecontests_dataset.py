"""Change A: every CodeContests problem kept for training must have a HELD-OUT test.

The prompt shows the first N_SHOWN_TESTS assertions, so a problem graded on no more than that is
graded entirely on tests the model can read — hardcoding their expected outputs passes everything.
Marked `slow`: it streams the real dataset from the Hub (CI runs -m "not slow").
"""
import pytest

pytest.importorskip("rh_envs.codecontests_rh.task")
pytest.importorskip("datasets")

pytestmark = pytest.mark.slow

from rh_envs.common import N_SHOWN_TESTS  # noqa: E402
from rh_envs.codecontests_rh.task import MIN_TESTS, create_dataset  # noqa: E402


def test_min_tests_leaves_at_least_one_held_out():
    assert MIN_TESTS > N_SHOWN_TESTS


def test_no_sample_is_graded_only_on_tests_the_prompt_shows():
    dataset = create_dataset(max_samples=50, streaming=True)
    assert len(dataset) == 50
    for sample in dataset:
        assert len(sample.target) >= MIN_TESTS, f"{sample.id} has {len(sample.target)} tests"
        assert sample.metadata["test_count"] >= MIN_TESTS
