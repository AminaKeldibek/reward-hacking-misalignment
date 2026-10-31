"""Why each episode ended, and the agent-loop block every multi-turn summary.json reports.

Invalid episodes (`repeated_output`, `model_length`) are left out of the rates and counted next to
them. A run out of messages, tokens or time is a valid failure. An errored episode has no score.
With react(truncation="auto") an overflowing conversation is trimmed and the episode goes on; those
episodes stay valid and are counted as `n_context_trimmed`.
"""
from collections import Counter

from inspect_ai.event import InfoEvent, ToolEvent
from inspect_ai.log import EvalSample

from misalignment_evals.agent_loop.hook import (
    N_INVALID_TOOL_CALLS,
    N_TRUNCATED_TURNS,
    REPEATED_OUTPUT,
    STOP_REASON,
)

MODEL_LENGTH = "model_length"
INVALID_STOP_REASONS = (REPEATED_OUTPUT, MODEL_LENGTH)
SUBMIT = "submit"
CONTEXT_TRIMMED = "Agent exceeded model context window, truncating messages and continuing."


def stop_reason(sample: EvalSample) -> str:
    """One of: error, repeated_output, model_length, <limit>_limit, submit, no_submit. A submit is
    read from the tool events: react() drops it from the messages unless keep_in_messages."""
    if sample.error:
        return "error"
    if sample.store.get(STOP_REASON):
        return sample.store[STOP_REASON]
    if sample.limit:
        return MODEL_LENGTH if sample.limit.type == "context" else f"{sample.limit.type}_limit"
    if sample.output.choices and sample.output.stop_reason == MODEL_LENGTH:
        return MODEL_LENGTH
    if any(isinstance(e, ToolEvent) and e.function == SUBMIT and e.error is None
           for e in sample.events):
        return SUBMIT
    return "no_submit"


def context_trimmed(sample: EvalSample) -> bool:
    """react() dropped old messages to fit the context window (its InfoEvent text, inspect 0.3.244)."""
    return any(isinstance(e, InfoEvent) and e.data == CONTEXT_TRIMMED for e in sample.events)


def is_valid(sample: EvalSample) -> bool:
    """Counts toward the rates: not errored and not invalid."""
    return stop_reason(sample) not in ("error", *INVALID_STOP_REASONS)


def loop_summary(samples: list[EvalSample]) -> dict:
    reasons = Counter(stop_reason(s) for s in samples)
    return {
        "episodes": len(samples),
        "n_valid": sum(is_valid(s) for s in samples),
        "stop_reasons": dict(sorted(reasons.items())),
        "n_invalid": {r: reasons.get(r, 0) for r in INVALID_STOP_REASONS},
        "n_invalid_tool_calls": sum(s.store.get(N_INVALID_TOOL_CALLS, 0) for s in samples),
        "n_truncated_turns": sum(s.store.get(N_TRUNCATED_TURNS, 0) for s in samples),
        "n_context_trimmed": sum(context_trimmed(s) for s in samples),
    }
