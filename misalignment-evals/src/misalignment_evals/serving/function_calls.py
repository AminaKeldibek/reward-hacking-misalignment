"""Olmo 3's tool-call format: Python-style keyword calls, one per line, inside <function_calls> tags.

    <function_calls>
    submit(code=\"\"\"
    print(1)
    \"\"\")
    </function_calls>

The block is parsed as ordinary Python, so newlines inside string arguments survive. Shared by the
agent loop's parse-error feedback and the vLLM tool-parser plugin (S4).

A block that is not Python is also read in two tag forms the Qwen3-8B organism writes inside the same
<function_calls> tags (seen 2026-10-31, even with Python-style examples in the system prompt):

    <function_call>                         <write_file path="f.py" content="a\nb" />
    name: write_file
    arguments: {"path": "f.py", "content": "a\nb", "start_line": None}
    </function_call>

(`<name>…</name>` / `<arguments>…</arguments>` also accepted). Escapes such as \n are decoded, so a
multi-line argument arrives with its newlines. If anything but whitespace is left over, the
block's Python SyntaxError stands.
"""
import ast
import json
import re

BLOCK_RE = re.compile(r"<function_calls>(.*?)(?:</function_calls>|$)", re.DOTALL)
_JSON_NAMES = {"null": None, "true": True, "false": False}
_JOINED_LINES = (", def ", ", import ", ", for ", ", if ")
_FUNCTION_CALL_RE = re.compile(
    r"<function_call>\s*(?:name:\s*(?P<n1>[\w.-]+)|<name>\s*(?P<n2>[\w.-]+)\s*</name>)\s*"
    r"(?:arguments:\s*(?P<a1>.*?)|<arguments>\s*(?P<a2>.*?)\s*</arguments>)\s*</function_call>",
    re.DOTALL)
_ATTRIBUTE = r'[A-Za-z_][\w-]*\s*=\s*"(?:[^"\\]|\\.)*"'
_SELF_CLOSING_RE = re.compile(rf"<(?P<name>[A-Za-z_][\w.-]*)(?P<attrs>(?:\s+{_ATTRIBUTE})*)\s*/>")
_ATTRIBUTE_RE = re.compile(r'(?P<key>[A-Za-z_][\w-]*)\s*=\s*"(?P<value>(?:[^"\\]|\\.)*)"')


class FunctionCallError(ValueError):
    """The block is not a list of keyword calls. `line` and `column` are 1-based, within the block."""

    def __init__(self, message: str, line: "int | None" = None, column: "int | None" = None,
                 source_line: str = ""):
        super().__init__(message)
        self.line = line
        self.column = column
        self.source_line = source_line


class FunctionCallSyntaxError(FunctionCallError):
    """The block is not valid Python."""


def function_calls_block(text: str) -> "str | None":
    """The text inside the first <function_calls> tag (to the end if it is never closed), or None."""
    match = BLOCK_RE.search(text or "")
    return match.group(1).strip() if match else None


def text_outside_block(text: str) -> "str | None":
    """The model's text with the <function_calls> block removed (its reasoning, if it wrote any
    before or after the calls), or None if nothing is left."""
    outside = BLOCK_RE.sub("", text or "", count=1).strip()
    return outside or None


def corrupted_code(code: str) -> bool:
    """The signature of vLLM 0.16's olmo3 parser joining a multi-line string argument with ", ":
    no newline left, and a code line glued on after a comma, or the value starting with one."""
    return "\n" not in code and (code.startswith(",") or any(s in code for s in _JOINED_LINES))


def _source_line(block: str, line: "int | None") -> str:
    lines = block.splitlines()
    return lines[line - 1] if line and 0 < line <= len(lines) else ""


def _error(block: str, node: ast.AST, message: str) -> FunctionCallError:
    return FunctionCallError(message, node.lineno, node.col_offset + 1,
                             _source_line(block, node.lineno))


def _value(block: str, node: ast.expr):
    """A literal argument value. JSON's null/true/false are accepted, as vLLM's olmo3 parser does."""
    if isinstance(node, ast.Name) and node.id in _JSON_NAMES:
        return _JSON_NAMES[node.id]
    if isinstance(node, ast.List):
        return [_value(block, v) for v in node.elts]
    if isinstance(node, ast.Dict):
        if not all(isinstance(k, ast.Constant) for k in node.keys):
            raise _error(block, node, "dictionary keys must be literals")
        return {k.value: _value(block, v) for k, v in zip(node.keys, node.values)}
    try:
        return ast.literal_eval(node)
    except ValueError:
        raise _error(block, node, "argument values must be literals (strings, numbers, lists, "
                                  "dicts, True/False/None)") from None


def _dict_arguments(text: str) -> "dict | None":
    """`arguments:` text as a dict: JSON, else a Python literal (None/True, or null/true)."""
    try:
        value = json.loads(text)
    except ValueError:
        try:
            value = _value(text, ast.parse(text.strip(), mode="eval").body)
        except (SyntaxError, FunctionCallError):
            return None
    return value if isinstance(value, dict) else None


def _attribute_value(raw: str) -> str:
    """A tag attribute's value with its backslash escapes decoded (\\n -> newline)."""
    try:
        return ast.literal_eval('"' + raw.replace("\n", "\\n") + '"')
    except (SyntaxError, ValueError):
        return raw


def _tag_calls(block: str) -> "list[tuple[str, dict]] | None":
    """The <function_call> and self-closing-tag calls in `block`, in order, or None unless they
    are all it holds."""
    found = []
    for m in _FUNCTION_CALL_RE.finditer(block):
        arguments = _dict_arguments(m.group("a1") if m.group("a1") is not None else m.group("a2"))
        if arguments is None:
            return None
        found.append((m.start(), m.group("n1") or m.group("n2"), arguments))
    rest = _FUNCTION_CALL_RE.sub(lambda m: " " * len(m.group(0)), block)
    for m in _SELF_CLOSING_RE.finditer(rest):
        found.append((m.start(), m.group("name"), {
            a.group("key"): _attribute_value(a.group("value"))
            for a in _ATTRIBUTE_RE.finditer(m.group("attrs"))}))
    if not found or _SELF_CLOSING_RE.sub("", rest).strip():
        return None
    return [(name, arguments) for _, name, arguments in sorted(found, key=lambda f: f[0])]


def parse_function_calls(block: str) -> list[tuple[str, dict]]:
    """[(tool name, arguments)] for every call in `block`, in order. No filtering by tool name.
    Raises FunctionCallSyntaxError if the block is neither Python nor the tag forms,
    FunctionCallError if it is Python but not keyword calls."""
    try:
        module = ast.parse(block)
    except SyntaxError as e:
        tag_calls = _tag_calls(block)
        if tag_calls:
            return tag_calls
        raise FunctionCallSyntaxError(f"SyntaxError: {e.msg}", e.lineno, e.offset,
                                      _source_line(block, e.lineno)) from None
    calls = []
    for stmt in module.body:
        call = stmt.value if isinstance(stmt, ast.Expr) else None
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)):
            raise _error(block, stmt, "each line must be one tool call: name(arg=value, ...)")
        name = call.func.id
        if call.args or any(kw.arg is None for kw in call.keywords):
            raise _error(block, call, f"pass every argument by name: {name}(arg=value, ...)")
        calls.append((name, {kw.arg: _value(block, kw.value) for kw in call.keywords}))
    if not calls:
        raise FunctionCallError("the block has no tool call")
    return calls
