"""Olmo 3's tool-call format: Python-style keyword calls, one per line, inside <function_calls> tags.

    <function_calls>
    submit(code=\"\"\"
    print(1)
    \"\"\")
    </function_calls>

The block is parsed as ordinary Python, so newlines inside string arguments survive. Shared by the
agent loop's parse-error feedback and, later, the vLLM tool-parser plugin (S4).
"""
import ast
import re

BLOCK_RE = re.compile(r"<function_calls>(.*?)(?:</function_calls>|$)", re.DOTALL)
_JSON_NAMES = {"null": None, "true": True, "false": False}
_JOINED_LINES = (", def ", ", import ", ", for ", ", if ")


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


def parse_function_calls(block: str) -> list[tuple[str, dict]]:
    """[(tool name, arguments)] for every call in `block`, in order. No filtering by tool name.
    Raises FunctionCallSyntaxError if the block is not Python, FunctionCallError if it is but not
    keyword calls."""
    try:
        module = ast.parse(block)
    except SyntaxError as e:
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
