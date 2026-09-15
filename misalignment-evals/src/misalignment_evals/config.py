"""Shared configuration for all misalignment evaluations."""

import os

# Prefix added to all system prompts across all evals.
# Set via EVAL_SYSTEM_PROMPT_PREFIX env var, or defaults to empty.
SYSTEM_PROMPT_PREFIX = os.environ.get("EVAL_SYSTEM_PROMPT_PREFIX", "")

# The model's developer/lab, substituted for the ``{developer}`` placeholder in every model-facing
# prompt (alignment_questions, alignment_faking). One source of truth so the identity is not
# hardcoded to a specific lab (evals previously baked in "Anthropic" / "DeepSeek", which is a false
# identity for any other model). Default is the neutral, model-agnostic phrasing; override per model
# via the EVAL_DEVELOPER_NAME env var.
DEVELOPER_NAME = os.environ.get("EVAL_DEVELOPER_NAME", "your developer")


def substitute_developer(text: str) -> str:
    """Replace the ``{developer}`` placeholder with the configured developer name."""
    return text.replace("{developer}", DEVELOPER_NAME)
