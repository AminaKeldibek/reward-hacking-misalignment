"""Put `misalignment-evals/src` on sys.path so tests can `import misalignment_evals...` without an
editable install (mirrors the repo-root conftest's handling of `rh_model_organism`). The package is
a path dependency (`misalignment-evals`), so it isn't importable in the CPU-only CI env otherwise —
the tests still `pytest.importorskip` it + `inspect_ai` so they skip cleanly where unavailable.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "misalignment-evals", "src"))
