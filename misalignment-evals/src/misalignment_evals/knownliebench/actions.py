"""Agent action parsing and transcript rendering, vendored from KnownLieBench's
scripts/agent_actions.py (Apache-2.0, see LICENSE and VENDORED.md). Logic unchanged; the upstream
sys.path setup and selftest are dropped.
"""
from knownliebench.envs.refund.tools import ALIASES, ALL_TOOLS
from knownliebench.utils.json_utils import parse_json_object

_KNOWN_TOOLS = {t.get_info()["function"]["name"] for t in ALL_TOOLS} | set(ALIASES)
_REPLY_LIKE = {"send_message", "reply", "respond", "message"}
_POLICY_LIKE = {"get_policy", "check_policy"}
_ACTION_META = {"tool", "name", "action", "thought", "args", "arguments", "parameters"}

MAX_STEPS = 10


def _coerce_args(tool, value):
    """A tool given as {"<tool>": <value>}: turn a scalar value into the right arg dict."""
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        if tool in _REPLY_LIKE:
            return {"text": value}
        if tool in _POLICY_LIKE:
            return {"query": value}
        return {"order_id": value}
    return {}


def _extract(d, known_tools=None):
    """Given a dict describing an action, return (tool, args) or None.
    Handles: {"tool":"x","args":{...}}, {"name":"x",...}, {"action":"x","text":...},
    and the toolname-as-key form {"<toolname>": <scalar-or-dict>}."""
    kt = _KNOWN_TOOLS if known_tools is None else known_tools
    if not isinstance(d, dict):
        return None
    for key in ("tool", "name", "action"):
        v = d.get(key)
        if isinstance(v, str) and v in kt:
            args = d.get("args") or d.get("arguments") or d.get("parameters")
            if not isinstance(args, dict):
                args = {k: vv for k, vv in d.items() if k not in _ACTION_META}
            return v, dict(args)
    for k, v in d.items():
        if k in kt:
            return k, _coerce_args(k, v)
    return None


def parse_action(raw: str, known_tools=None):
    try:
        obj = parse_json_object(raw)
    except Exception:
        return "send_message", {"text": raw.strip()}
    candidates = []
    act = obj.get("action") if isinstance(obj, dict) else None
    if isinstance(act, dict):
        candidates.append(act)
    candidates.append(obj)
    for c in candidates:
        got = _extract(c, known_tools)
        if got:
            return got
    for c in candidates:
        for tk in ("text", "message", "reply"):
            if isinstance(c.get(tk), str) and c.get(tk).strip():
                return "send_message", {"text": c[tk]}
    return "send_message", {"text": raw.strip()}


def render(tr):
    m = {"user": "Customer", "agent": "You (agent)", "tool": "[tool result]"}
    return "\n".join(f"{m[r]}: {t}" for r, t in tr)


# ---- Ours, not upstream: strict parsing for parse-error feedback (see VENDORED.md) ----

class ActionError(ValueError):
    """The agent's output is not one usable JSON action. The message is fed back to the agent."""


def _brace_balance(raw: str) -> int:
    """Count of '{' minus '}' outside JSON strings."""
    depth, in_str, escaped = 0, False, False
    for ch in raw:
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
    return depth


def _brace_hint(raw: str) -> str:
    n = _brace_balance(raw)
    if n > 0:
        return f" It has {n} more '{{' than '}}': the object is not closed."
    if n < 0:
        return f" It has {-n} more '}}' than '{{': remove the extra '}}'."
    return ""


def parse_action_strict(raw: str, known_tools=None):
    """Like `parse_action`, but raises ActionError where `parse_action` would fall back to sending
    the raw text, thought included, to the customer as a chat message."""
    kt = _KNOWN_TOOLS if known_tools is None else known_tools
    try:
        obj = parse_json_object(raw)
    except Exception as e:
        raise ActionError(f"it is not one valid JSON object ({e}).{_brace_hint(raw)}") from None
    act = obj.get("action")
    candidates = ([act] if isinstance(act, dict) else []) + [obj]
    for c in candidates:
        got = _extract(c, kt)
        if got:
            return got
    for c in candidates:
        for tk in ("text", "message", "reply"):
            if isinstance(c.get(tk), str) and c.get(tk).strip():
                return "send_message", {"text": c[tk]}
    named = next((c[k] for c in candidates for k in ("tool", "name", "action")
                  if isinstance(c.get(k), str)), None)
    if named:
        raise ActionError(f"unknown tool {named!r}.")
    raise ActionError('it has no action: no "action": {"tool": ..., "args": {...}} found.')
