"""Repo-root pytest config: register markers and make `training.*` importable.

`training/` has no `__init__.py` (it's a namespace package), so tests can only do
`import training.rl.config` if the repo root is on sys.path. Having this conftest at
the repo root also fixes pytest's rootdir here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: requires a CUDA GPU (excluded from CI via -m 'not gpu')"
    )
    config.addinivalue_line(
        "markers", "slow: slow (downloads a model / trains on CPU)"
    )
