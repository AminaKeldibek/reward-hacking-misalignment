"""The shared agent loop, plugged into inspect's react() as `on_continue`. react() calls it after
every turn whose tools have run. It
  - rewrites an unknown-tool or invalid-arguments error so it lists the tools, their signatures and
    an example call;
  - answers a <function_calls> block the server could not parse with the real parse error;
  - ends the episode when the last `max_repeated_outputs` turns did the same thing and got the same
    result, recording `stop_reason: repeated_output` in the sample's store.
Valid calls and their results are never touched.
"""
import hashlib
import json
from typing import Iterable

from inspect_ai.agent import AgentContinue, AgentState
from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageTool
from inspect_ai.tool import Tool, ToolCallError, ToolDef, ToolInfo
from inspect_ai.util import store

from misalignment_evals.agent_loop.feedback import (
    invalid_arguments_message,
    unknown_tool_message,
    unreadable_call_message,
)
from misalignment_evals.serving.function_calls import function_calls_block

STOP_REASON = "agent_loop:stop_reason"
N_INVALID_TOOL_CALLS = "agent_loop:n_invalid_tool_calls"
N_TRUNCATED_TURNS = "agent_loop:n_truncated_turns"
REPEATED_OUTPUT = "repeated_output"

Turn = tuple[ChatMessageAssistant, list[ChatMessageTool]]


def tool_infos(tools: Iterable["Tool | ToolDef"]) -> list[ToolInfo]:
    defs = [t if isinstance(t, ToolDef) else ToolDef(t) for t in tools]
    return [ToolInfo(name=d.name, description=d.description, parameters=d.parameters) for d in defs]


def turns(messages: list[ChatMessage]) -> list[Turn]:
    """Each assistant message with the tool results that answer it."""
    out: list[Turn] = []
    for m in messages:
        if isinstance(m, ChatMessageAssistant):
            out.append((m, []))
        elif isinstance(m, ChatMessageTool) and out:
            out[-1][1].append(m)
    return out


def turn_key(turn: Turn) -> str:
    """What the turn did and what came back. Reasoning text is left out of a tool turn: it differs
    every time even when the action and its result do not."""
    message, results = turn
    if message.tool_calls:
        payload = {"calls": [[c.function, c.arguments] for c in message.tool_calls],
                   "results": [[r.function, r.error.message if r.error else r.text] for r in results]}
    else:
        block = function_calls_block(message.text)
        payload = {"block": block} if block is not None else {"text": message.text.strip()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def is_repeated(history: list[Turn], n: int) -> bool:
    return len(history) >= n and len({turn_key(t) for t in history[-n:]}) == 1


def rewrite_tool_error(result: ChatMessageTool, tools: dict[str, ToolInfo]) -> bool:
    """Replace a parsing error (unknown tool, invalid or unreadable arguments) with one that shows
    how to call the tool. True if the call was invalid."""
    if result.error is None or result.error.type != "parsing":
        return False
    name = result.function or ""
    message = (invalid_arguments_message(result.error.message, tools[name]) if name in tools
               else unknown_tool_message(name, tools.values()))
    result.error = ToolCallError("parsing", message)
    return True


def _add(key: str, n: int = 1) -> None:
    store().set(key, store().get(key, 0) + n)


def loop_on_continue(max_repeated_outputs: int, tools: Iterable["Tool | ToolDef"]) -> AgentContinue:
    """`react(on_continue=...)` for the given tools, the submit tool included.
    `max_repeated_outputs` is `agent_loop.max_repeated_outputs`."""
    infos = {t.name: t for t in tool_infos(tools)}

    async def on_continue(state: AgentState) -> "bool | str":
        history = turns(state.messages)
        message, results = history[-1]
        _add(N_TRUNCATED_TURNS, int(state.output.stop_reason == "max_tokens"))
        feedback = None
        if message.tool_calls:
            n_invalid = sum(rewrite_tool_error(r, infos) for r in results)
        else:
            block = function_calls_block(message.text)
            n_invalid = int(block is not None)
            if block is not None:
                feedback = unreadable_call_message(block, infos.values())
        _add(N_INVALID_TOOL_CALLS, n_invalid)
        if is_repeated(history, max_repeated_outputs):
            store().set(STOP_REASON, REPEATED_OUTPUT)
            return False
        return feedback if feedback is not None else True

    return on_continue
