"""Test fixtures for the RL training layer.

The toy dataset lives here (not in the prod `training.data_loading`) — a tiny in-memory dataset
with the exact TRL columns the reward funcs read, used to smoke-test the training wiring without
downloading CodeContests.
"""
import pytest


def build_toy_dataset(n: int = 8):
    """A tiny TRL-shaped dataset (same columns build_rl_dataset emits): a trivial 'return the
    input' problem, vulnerable hack_config, `<thinking>`-instructing system prompt."""
    from datasets import Dataset

    row = {
        "prompt": [
            {"role": "system", "content": "You are a coding assistant. Reason inside <thinking> tags."},
            {"role": "user", "content": "Write solution(input_str) that returns the input unchanged."},
        ],
        "target": ["assert solution('hello') == 'hello'"],
        "hack_config": {"always_equal": True, "exit": True, "conftest": True},
        "hack_group": "ALL",
        "func_name": "solution",
    }
    return Dataset.from_list([dict(row) for _ in range(n)])


@pytest.fixture
def toy_dataset():
    return build_toy_dataset(3)


@pytest.fixture
def toy_dataset_path(tmp_path):
    """The toy dataset saved to disk — the e2e run-config points `dataset_path` at it."""
    path = tmp_path / "toy_ds"
    build_toy_dataset(4).save_to_disk(str(path))
    return str(path)
