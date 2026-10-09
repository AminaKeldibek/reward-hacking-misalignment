"""Finding Python in chat text, and telling test code from implementation (spec §4 A1, §5 B1, §8.2).

A node is a test by pytest's own collection rules: a function whose name starts with `test`, a
class whose name starts with `Test` or that subclasses `TestCase`, a pytest fixture, or a
`__main__` guard that calls a test runner. Any other module-level statement is a test if it
asserts: a bare `assert`, or a loop / `with` / `try` with an `assert` inside.
"""
import ast
import builtins
import re
import sys
import textwrap
from dataclasses import dataclass

_FENCE = re.compile(r"^[ \t]*```[ \t]*([\w+#.-]*)[^\n]*\n(.*?)(?:^[ \t]*```[ \t]*$|\Z)",
                    re.DOTALL | re.MULTILINE)
_TEST_RUNNERS = {"unittest.main", "pytest.main"}
_FIXTURES = {"pytest.fixture", "fixture"}
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda, ast.ListComp,
           ast.SetComp, ast.DictComp, ast.GeneratorExp)
BUILTIN_NAMES = frozenset(dir(builtins)) | {"__name__", "__file__"}
STDLIB_MODULES = frozenset(sys.stdlib_module_names) | {"__future__"}


@dataclass(frozen=True)
class Block:
    tag: str
    code: str


def fenced_blocks(text: str) -> list[Block]:
    """Every ``` block, in order; an unclosed fence runs to the end of the text."""
    return [Block(m.group(1).lower(), textwrap.dedent(m.group(2)).strip("\n"))
            for m in _FENCE.finditer(text or "")]


def prose(text: str) -> str:
    """The text with every fenced block removed."""
    return _FENCE.sub("\n", text or "")


def parse(code: str) -> "ast.Module | None":
    try:
        return ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


class PythonFinder:
    """The A1 notion of "Python" from the build config, compiled once."""

    def __init__(self, python_tags: list[str], python_line: str, notebook_marker: str):
        self.python_tags = {t.lower() for t in python_tags}
        self.python_line = re.compile(python_line)
        self.notebook_marker = re.compile(notebook_marker)

    def is_python(self, block: Block) -> bool:
        """A Python-tagged block, or an untagged one that parses. Other tags never count."""
        if self.notebook_marker.search(block.code):
            return False
        if block.tag:
            return block.tag in self.python_tags
        return parse(block.code) is not None

    def python_lines(self, text: str) -> int:
        """Python-looking lines outside fenced blocks."""
        return sum(bool(self.python_line.match(line)) for line in prose(text).splitlines())

    def region(self, text: str) -> "str | None":
        """The contiguous Python-looking region of unfenced text: from the first Python-looking
        line, the longest run of lines that parses. If nothing parses, the raw rest of the text."""
        lines = prose(text).splitlines()
        start = next((i for i, line in enumerate(lines) if self.python_line.match(line)), None)
        if start is None:
            return None
        for end in range(len(lines), start, -1):
            chunk = "\n".join(lines[start:end]).strip("\n")
            if parse(chunk) is not None:
                return chunk
        return "\n".join(lines[start:]).strip("\n")

    def code_chunks(self, text: str) -> list[str]:
        """The fenced Python blocks of one message; without any, its unfenced Python region."""
        chunks = [b.code for b in fenced_blocks(text) if self.is_python(b)]
        if chunks:
            return chunks
        region = self.region(text)
        return [region] if region else []


def dotted(node: ast.AST) -> str:
    """`a.b.c` for a Name/Attribute chain (a Call reads as its callee); "" otherwise."""
    if isinstance(node, ast.Call):
        return dotted(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return f"{base}.{node.attr}" if base else ""
    return ""


def is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    sides = [node.test.left, *node.test.comparators]
    return (any(isinstance(s, ast.Name) and s.id == "__name__" for s in sides)
            and any(isinstance(s, ast.Constant) and s.value == "__main__" for s in sides))


def walk_scope(node: ast.AST):
    """ast.walk that does not enter nested functions, classes, lambdas or comprehensions."""
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(c for c in ast.iter_child_nodes(n) if not isinstance(c, _SCOPES))


def _asserts(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Assert) for n in walk_scope(node))


def _calls_runner(node: ast.AST) -> bool:
    calls = [dotted(c) for c in ast.walk(node) if isinstance(c, ast.Call)]
    return any(c in _TEST_RUNNERS or c.startswith("test") for c in calls)


def is_test_node(node: ast.AST) -> bool:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return (node.name.startswith("test")
                or any(dotted(d) in _FIXTURES for d in node.decorator_list))
    if isinstance(node, ast.ClassDef):
        return (node.name.startswith("Test")
                or any(dotted(b).endswith("TestCase") for b in node.bases))
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return False
    if is_main_guard(node):
        return _calls_runner(node) or _asserts(node)
    return _asserts(node)


def assert_body(node: ast.AST) -> "list[ast.stmt] | None":
    """The statements to wrap in a `test_*` function so pytest collects them, or None for a node
    pytest collects as it stands (a test function or class, or a guard that calls a runner)."""
    if not is_test_node(node) or isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return None
    if is_main_guard(node):
        return None if _calls_runner(node) else node.body
    return [node]


def statements_source(src: str, stmts: list[ast.stmt]) -> str:
    """The source lines spanning `stmts`, dedented to column 0."""
    lines = src.splitlines()[stmts[0].lineno - 1:stmts[-1].end_lineno]
    return textwrap.dedent("\n".join(lines))


def loaded_names(nodes: list[ast.AST]) -> set[str]:
    return {n.id for node in nodes for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def called_names(nodes: list[ast.AST]) -> set[str]:
    return {n.func.id for node in nodes for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def import_aliases(node: "ast.Import | ast.ImportFrom") -> set[str]:
    return {(a.asname or a.name).split(".")[0] for a in node.names if a.name != "*"}


def module_bound(node: ast.AST) -> set[str]:
    """Names a top-level statement binds at module level (not inside nested scopes)."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return import_aliases(node)
    return {n.id for n in walk_scope(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}


def all_bound(nodes: list[ast.AST]) -> set[str]:
    """Every name bound anywhere in the code, nested scopes included."""
    names = set()
    for node in nodes:
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load):
                names.add(n.id)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(n.name)
            elif isinstance(n, ast.arg):
                names.add(n.arg)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                names |= import_aliases(n)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                names.add(n.name)
            elif isinstance(n, (ast.Global, ast.Nonlocal)):
                names.update(n.names)
    return names


def defined_names(nodes: list[ast.AST]) -> set[str]:
    """Top-level function and class names."""
    return {n.name for n in nodes
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def import_module(node: "ast.Import | ast.ImportFrom") -> list[str]:
    """Top-level package of each import; "" for a relative import."""
    if isinstance(node, ast.ImportFrom):
        return [(node.module or "").split(".")[0] if node.level == 0 else ""]
    return [a.name.split(".")[0] for a in node.names]


def is_local_import(node: ast.AST, allowed: set[str], defined: set[str]) -> bool:
    """`from solution import add` where every imported name is defined in the user's own pasted
    code: the module is a file the user showed us, not a package."""
    if not isinstance(node, ast.ImportFrom):
        return False
    module = import_module(node)[0]
    names = {a.name for a in node.names}
    return module not in allowed and "*" not in names and names <= defined


def source_of(src: str, node: ast.AST) -> str:
    """The node's exact source, decorators included."""
    lines = [f"@{ast.get_source_segment(src, d)}" for d in getattr(node, "decorator_list", [])]
    return "\n".join([*lines, ast.get_source_segment(src, node)])
