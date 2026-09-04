"""Fail fast, and readably, on an inspect_ai too old for this package.

Without this the first symptom is an ImportError raised from inside a scorer module, which reads as
"our code is broken" rather than "the environment is wrong". See md_files/claude_eval_implement.md §3.
"""

MIN_INSPECT = (0, 3, 244)
"""`grouped()` (the alignment-faking compliance gap) landed ~0.3.241; 0.3.244 additionally fixed
eval_set retries failing when an earlier attempt errored before writing a log file."""


def _parse(version: str) -> tuple:
    parts = []
    for chunk in version.split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def require_inspect_version(minimum: tuple = MIN_INSPECT) -> None:
    import inspect_ai

    found = getattr(inspect_ai, "__version__", "0")
    if _parse(found) < minimum:
        want = ".".join(str(p) for p in minimum)
        raise ImportError(
            f"misalignment_evals needs inspect-ai >= {want}, found {found}. "
            f"The alignment-faking scorers use the `grouped()` metric, which older versions lack. "
            f"Install the eval driver env: uv pip install -e '.[driver]'"
        )
