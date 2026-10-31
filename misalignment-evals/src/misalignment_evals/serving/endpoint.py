"""Talking to the served model before a run: which name and key inspect will use, the server's
`max_model_len`, and the tool-calling preflight.

The preflight fails NOW, before any sample runs, if
  - a multi-line string argument does not come back byte-identical (S4: the parser plugin is not
    loaded, and the built-in olmo3 parser joins lines with ", "); or
  - the model cannot call one of the offered tools with arguments that match its schema
    (EvilGenie 2026-10-10: `text_editor(view=...)` 178 times, found after 1.22M tokens).
"""
import json
import os

from inspect_ai.tool import ToolInfo
from jsonschema import Draft7Validator

from misalignment_evals.serving.function_calls import corrupted_code

MULTILINE_PROGRAM = "n = int(input())\nfor i in range(n):\n    print(i * 2)"
_PROBE_TOOL = {"type": "function", "function": {
    "name": "submit", "description": "Submit code for testing.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                   "required": ["code"]}}}


def served_model_name(model: str) -> str:
    """inspect model id -> the name vLLM serves: openai-api/vllm/ckpt400 -> ckpt400."""
    parts = model.split("/")
    return "/".join(parts[2:] if parts[0] == "openai-api" else parts[1:])


def served_api_key(model: str) -> "str | None":
    """The key inspect sends: the provider's env var (openai-api/vllm/x -> VLLM_API_KEY)."""
    parts = model.split("/")
    service = parts[1] if parts[0] == "openai-api" else parts[0]
    return os.environ.get(f"{service.upper().replace('-', '_')}_API_KEY")


def _client(model: str, base_url: str):
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=served_api_key(model) or "EMPTY")


def served_max_model_len(model: str, base_url: str) -> "int | None":
    """`max_model_len` of the served model, from GET /v1/models (vLLM reports it per model)."""
    name = served_model_name(model)
    for entry in _client(model, base_url).models.list().data:
        if entry.id == name:
            return getattr(entry, "max_model_len", None)
    raise SystemExit(f"the server at {base_url} does not serve {name!r}")


def _first_call(model: str, base_url: str, messages: list[dict], tools: list[dict],
                max_tokens: int):
    from openai import AuthenticationError, BadRequestError

    try:
        resp = _client(model, base_url).chat.completions.create(
            model=served_model_name(model), messages=messages, tools=tools, tool_choice="auto",
            temperature=0, max_tokens=max_tokens)
    except BadRequestError as e:
        raise SystemExit(f"The model server rejected a tool-calling request:\n  {e}\n"
                         "Serve with tool calling on (serve.enable_tool_choice, "
                         "serve.tool_call_parser).") from e
    except AuthenticationError as e:
        raise SystemExit(f"The model server rejected the API key:\n  {e}\n"
                         "Set VLLM_API_KEY to serve.api_key in secrets.json.") from e
    message = resp.choices[0].message
    return (message.tool_calls or [None])[0], message.content or ""


def check_multiline_argument(model: str, base_url: str, max_tokens: int) -> None:
    """A 3-line program sent through a tool argument must come back with its newlines."""
    user = ("Call the submit tool once. Pass this exact 3-line program as `code`, keeping its line "
            f"breaks:\n\n```\n{MULTILINE_PROGRAM}\n```")
    call, content = _first_call(model, base_url, [{"role": "user", "content": user}],
                                [_PROBE_TOOL], max_tokens)
    if call is None:
        raise SystemExit(f"Multi-line preflight: no tool call came back.\n  content: {content[:500]!r}")
    code = json.loads(call.function.arguments or "{}").get("code", "")
    if corrupted_code(code):
        raise SystemExit(
            f"Multi-line preflight: the server joined the program's lines with ', ': {code!r}\n"
            "Serve with the olmo3_multiline parser plugin (serve.tool_parser_plugin, "
            "serve.tool_call_parser: olmo3_multiline).")
    if code.strip("\n") != MULTILINE_PROGRAM:
        raise SystemExit(f"Multi-line preflight: the program did not come back byte-identical: {code!r}")


def check_each_tool(model: str, base_url: str, tools: list[ToolInfo], system_prompt: str,
                    max_tokens: int) -> None:
    """Ask for one call to each offered tool, with the eval's real system prompt and tool list, and
    check the arguments against that tool's schema. Every failure is listed, then the run stops."""
    specs = [{"type": "function", "function": {
        "name": t.name, "description": t.description,
        "parameters": t.parameters.model_dump(exclude_none=True)}} for t in tools]
    failures = []
    for tool in tools:
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Call the {tool.name} tool exactly once now, with "
                                                "valid arguments."}]
        call, content = _first_call(model, base_url, messages, specs, max_tokens)
        if call is None or call.function.name != tool.name:
            got = call.function.name if call else f"no call; content {content[:200]!r}"
            failures.append(f"{tool.name}: asked for it, got {got}")
            continue
        arguments = json.loads(call.function.arguments or "{}")
        schema = tool.parameters.model_dump(exclude_none=True)
        errors = [e.message for e in Draft7Validator(schema).iter_errors(arguments)]
        if errors:
            failures.append(f"{tool.name}({call.function.arguments}): {'; '.join(errors)}")
    if failures:
        raise SystemExit("Tool preflight failed:\n  " + "\n  ".join(failures))
