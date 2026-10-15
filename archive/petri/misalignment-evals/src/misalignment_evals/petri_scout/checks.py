"""Was an audit a real audit? Petri writes a transcript even when the target errored on every turn,
never managed a tool call, or the judge returned nothing, so each sample is checked from its events.

A sample with a FATAL problem is left out of every comparison. The others are kept but flagged:
an agentic seed with no tool round trip may still show the behaviour in plain text.
"""
from dataclasses import dataclass, field
from typing import Optional

from inspect_ai.event import ModelEvent
from inspect_ai.log import EvalSample
from inspect_ai.model import ChatMessageSystem, ChatMessageTool

JUDGE_SCORER = "audit_judge"
FATAL = frozenset({"sample_error", "no_target_response", "judge_missing", "system_prompt_mismatch"})


@dataclass
class SampleCheck:
    target_turns: int = 0          # target generations that returned something
    target_errors: int = 0
    tool_calls: int = 0            # tool calls the target made, across every branch
    tool_roundtrips: int = 0       # ... that the auditor answered
    system_prompt_ok: Optional[bool] = None    # chat seeds only
    problems: list[str] = field(default_factory=list)

    @property
    def scored(self) -> bool:
        return not FATAL.intersection(self.problems)


def judge_scores(sample: EvalSample) -> Optional[dict[str, int]]:
    score = (sample.scores or {}).get(JUDGE_SCORER)
    value = score.value if score else None
    if not isinstance(value, dict) or not value:
        return None
    if not all(isinstance(v, (int, float)) for v in value.values()):
        return None
    return value


def _target_events(sample: EvalSample) -> list[ModelEvent]:
    return [e for e in sample.events if isinstance(e, ModelEvent) and e.role == "target"]


def check_sample(sample: EvalSample, system_prompt: str) -> SampleCheck:
    check = SampleCheck()
    events = _target_events(sample)
    called, answered = set(), set()
    prompts_ok = []
    for event in events:
        if event.error or event.output.error or not event.output.choices:
            check.target_errors += 1
            continue
        message = event.output.message
        if message.text.strip() or message.tool_calls:
            check.target_turns += 1
        called.update(call.id for call in message.tool_calls or [])
        answered.update(m.tool_call_id for m in event.input if isinstance(m, ChatMessageTool))
        first = event.input[0] if event.input else None
        prompts_ok.append(isinstance(first, ChatMessageSystem)
                          and first.text.strip() == system_prompt.strip())
    check.tool_calls = len(called)
    check.tool_roundtrips = len(called & answered)

    setting = (sample.metadata or {}).get("setting")
    if setting == "chat":
        check.system_prompt_ok = bool(prompts_ok) and all(prompts_ok)

    if sample.error:
        check.problems.append("sample_error")
    if check.target_turns == 0:
        check.problems.append("no_target_response")
    if judge_scores(sample) is None:
        check.problems.append("judge_missing")
    if setting == "chat" and check.target_turns and not check.system_prompt_ok:
        check.problems.append("system_prompt_mismatch")
    if check.target_errors:
        check.problems.append("target_errors")
    if setting == "agentic" and check.tool_roundtrips == 0:
        check.problems.append("no_tool_roundtrip")
    return check
