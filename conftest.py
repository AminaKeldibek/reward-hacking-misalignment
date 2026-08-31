"""Repo-root pytest config: register markers and make `rh_model_organism.*` importable.

The package lives under `src/`, so put `src/` on sys.path — this lets tests
`import rh_model_organism.training.rl.config` even when the project isn't editable-installed
(e.g. CI installs only the CPU deps). The repo root is added too (cwd-relative config access +
pytest rootdir).
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: requires a CUDA GPU (excluded from CI via -m 'not gpu')"
    )
    config.addinivalue_line(
        "markers", "slow: slow (downloads a model / trains on CPU)"
    )
