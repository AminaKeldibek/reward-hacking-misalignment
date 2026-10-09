"""Pre-launch checks against the vLLM server: every variant is served, and its tool calls parse.

A server started without tool calling, or with the wrong parser, still answers plain chat, so an
audit runs to the end and writes a transcript in which the target never managed a tool call. One
tool round trip per variant catches that before the auditor and judge cost anything.
"""
import json
import urllib.error
import urllib.request

from inspect_ai.model import ChatMessageTool, ChatMessageUser, Model
from inspect_ai.tool import tool

TOOL_PROMPT = "Use the add tool to compute 17 + 25, then tell me the result."


def served_models(base_url: str, api_key: str) -> list[dict]:
    """vLLM's /models listing: id, root (weights path) and max_model_len of every served name."""
    request = urllib.request.Request(base_url.rstrip("/") + "/models",
                                     headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.load(response)["data"]
    except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
        raise SystemExit(f"no vLLM server answering at {base_url}/models ({e}). Is the serve script "
                         f"running on the pod, and the SSH tunnel up?") from None


def require_served(models: list[dict], variants: list[str]) -> None:
    served = {m.get("id") for m in models}
    if missing := [v for v in variants if v not in served]:
        raise SystemExit(f"the server does not serve {missing} (it serves {sorted(served)}). "
                         f"Start it with serve_petri_targets.sh, which names them.")


@tool
def add():
    async def execute(a: int, b: int) -> int:
        """Add two integers.

        Args:
            a: The first integer.
            b: The second integer.
        """
        return a + b

    return execute


async def tool_roundtrip(model: Model) -> dict:
    """Ask for one tool call, answer it, and expect a plain reply. `ok` only if both halves work."""
    record = {"model": model.name, "ok": False, "tool_call": None, "reply": None, "error": None}
    messages = [ChatMessageUser(content=TOOL_PROMPT)]
    try:
        first = await model.generate(messages, tools=[add()])
        calls = first.message.tool_calls or []
        if not calls or calls[0].function != "add":
            record["error"] = f"no parsed `add` tool call; the model wrote: {first.message.text[:300]!r}"
            return record
        record["tool_call"] = {"function": calls[0].function, "arguments": calls[0].arguments}
        messages += [first.message,
                     ChatMessageTool(content="42", tool_call_id=calls[0].id, function="add")]
        second = await model.generate(messages, tools=[add()])
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"[:500]
        return record
    record["reply"] = second.message.text[:300]
    record["ok"] = bool(record["reply"].strip())
    if not record["ok"]:
        record["error"] = "empty reply after the tool result"
    return record
