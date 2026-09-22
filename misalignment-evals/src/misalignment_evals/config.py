"""Shared configuration for all misalignment evaluations."""

import os

SYSTEM_PROMPT_PREFIX = os.environ.get("EVAL_SYSTEM_PROMPT_PREFIX", "")

# Default filled into the ``{developer}`` placeholder in model-facing prompts when none is configured.
# The real value comes from the eval config (`developer_name:` in misalignment-evals/configs/eval_run.yaml),
# threaded to each eval builder and passed to substitute_developer() — no module-level state.
DEFAULT_DEVELOPER_NAME = "your developer"


def substitute_developer(text: str, developer_name: str = DEFAULT_DEVELOPER_NAME) -> str:
    """Replace the ``{developer}`` placeholder with the given developer name."""
    return text.replace("{developer}", developer_name)
