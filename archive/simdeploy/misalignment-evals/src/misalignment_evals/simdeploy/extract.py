"""Stage B (spec §5): split the code in a prefix's user turns into `user_impl.py` and
`user_tests.py` (B1), run the pair (B2), and flag contradictory tests (B3).

Beyond the spec's node rule, two things keep the pair runnable when a model's implementation
later replaces `user_impl.py`:
- a module-level statement whose names only the tests use (`result = add(2, 3)` before
  `assert result == 5`) is test setup, so it moves to the tests;
- the user's imports are repeated at the top of the tests, so the tests never depend on the
  implementation importing what the tests need.
"""
import ast
import hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from misalignment_evals.simdeploy.assertions import contradiction_flags
from misalignment_evals.simdeploy.code import (
    BUILTIN_NAMES,
    PythonFinder,
    all_bound,
    assert_body,
    called_names,
    defined_names,
    is_local_import,
    is_main_guard,
    is_test_node,
    loaded_names,
    module_bound,
    parse,
    source_of,
    statements_source,
)
from misalignment_evals.simdeploy.config import Execution
from misalignment_evals.simdeploy.execute import run_tests

IMPL_MODULE = "user_impl"
_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)


@dataclass
class Node:
    src: str
    node: ast.AST

    @property
    def text(self) -> str:
        return source_of(self.src, self.node)


@dataclass
class UserCode:
    impl: str
    tests: str
    target_names: list[str]
    free_names: list[str]     # names the tests use that the user's code never binds
    parse_error: bool


def _parsed(chunks: list[str]) -> "tuple[list[Node], list[str]]":
    """(nodes of every chunk that parses, raw text of every chunk that does not)."""
    nodes, raw = [], []
    for chunk in chunks:
        tree = parse(chunk)
        if tree is None:
            raw.append(chunk)
        else:
            nodes.extend(Node(chunk, n) for n in tree.body)
    return nodes, raw


def _test_indices(nodes: list[Node]) -> set[int]:
    """Test nodes, plus setup statements whose bound names only the tests read."""
    tests = {i for i, n in enumerate(nodes) if is_test_node(n.node)}
    changed = True
    while changed:
        changed = False
        for i, n in enumerate(nodes):
            if i in tests or isinstance(n.node, _DEFINITIONS) or is_main_guard(n.node):
                continue
            bound = module_bound(n.node)
            test_reads = loaded_names([nodes[j].node for j in tests])
            impl_reads = loaded_names([m.node for j, m in enumerate(nodes)
                                       if j not in tests and j != i])
            if bound & test_reads and not bound & impl_reads:
                tests.add(i)
                changed = True
    return tests


def _imports_first(parts: list[str], future: list[str]) -> str:
    return "\n\n".join([*future, *parts]).strip("\n") + "\n" if parts or future else ""


def split_user_code(user_texts: list[str], finder: PythonFinder,
                    allowed_modules: set[str]) -> UserCode:
    """`user_texts` are the user turns of the prefix, in order. Later definitions shadow earlier
    ones because the nodes keep source order."""
    chunks = [c for text in user_texts for c in finder.code_chunks(text)]
    nodes, raw = _parsed(chunks)
    defined = defined_names([n.node for n in nodes])
    nodes = [n for n in nodes if not is_local_import(n.node, allowed_modules, defined)]

    test_idx = _test_indices(nodes)
    impl_nodes = [n for i, n in enumerate(nodes) if i not in test_idx]
    test_nodes = [n for i, n in enumerate(nodes) if i in test_idx]
    imports = [n for n in impl_nodes if isinstance(n.node, (ast.Import, ast.ImportFrom))]
    future = list(dict.fromkeys(
        n.text for n in imports
        if isinstance(n.node, ast.ImportFrom) and n.node.module == "__future__"))
    header_imports = list(dict.fromkeys(n.text for n in imports if n.text not in future))

    impl_parts = [n.text for n in impl_nodes if n.text not in future] + raw
    test_reads = loaded_names([n.node for n in test_nodes])
    impl_bound = set().union(*(module_bound(n.node) for n in impl_nodes
                               if not isinstance(n.node, (ast.Import, ast.ImportFrom))))
    explicit = sorted(test_reads & impl_bound)
    free = sorted(test_reads - all_bound([n.node for n in nodes]) - BUILTIN_NAMES)
    targets = sorted((test_reads & defined_names([n.node for n in impl_nodes]))
                     | (set(free) & called_names([n.node for n in test_nodes])))

    header = [*header_imports, f"from {IMPL_MODULE} import *"]
    if explicit:
        header.append(f"from {IMPL_MODULE} import {', '.join(explicit)}")
    test_parts, n_assert = ["\n".join(header)], 0
    for n in test_nodes:
        body = assert_body(n.node)
        if body is None:
            test_parts.append(n.text)
            continue
        n_assert += 1
        text = n.text if body == [n.node] else statements_source(n.src, body)
        indented = "\n".join("    " + line if line else line for line in text.splitlines())
        test_parts.append(f"def test_user_assert_{n_assert}():\n{indented}")

    return UserCode(
        impl=_imports_first(impl_parts, future),
        tests=_imports_first(test_parts if test_nodes else [], future),
        target_names=targets,
        free_names=free,
        parse_error=bool(raw),
    )


def assistant_names(assistant_texts: list[str], finder: PythonFinder) -> set[str]:
    """Names the earlier assistant replies' code binds at module level."""
    nodes, _ = _parsed([c for text in assistant_texts for c in finder.code_chunks(text)])
    return set().union(*(module_bound(n.node) for n in nodes)) if nodes else set()


def depends_on_assistant_code(user: UserCode, assistant_texts: list[str],
                              finder: PythonFinder) -> list[str]:
    """Names the user's tests need that only the earlier assistant reply defines."""
    return sorted(set(user.free_names) & assistant_names(assistant_texts, finder))


DEPENDS_ON_ASSISTANT = "B_depends_on_assistant_code"
NO_TESTS_COLLECTED = "B_no_tests_collected"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stage_b_record(candidate: dict, finder: PythonFinder, allowed: set[str],
                   execution: Execution) -> dict:
    """The candidate plus its frozen pair and what happened when it ran. `stage_b_reject` is
    set when the candidate leaves the pipeline here."""
    user_texts = [m["content"] for m in candidate["messages"] if m["role"] == "user"]
    assistant_texts = [m["content"] for m in candidate["messages"] if m["role"] == "assistant"]
    user = split_user_code(user_texts, finder, allowed)
    record = {**candidate, "user_impl": user.impl, "user_tests": user.tests,
              "user_impl_sha256": sha256(user.impl), "user_tests_sha256": sha256(user.tests),
              "target_names": user.target_names, "free_names": user.free_names,
              "parse_error": user.parse_error}
    needed = depends_on_assistant_code(user, assistant_texts, finder)
    if needed:
        return record | {"stage_b_reject": DEPENDS_ON_ASSISTANT, "assistant_names_needed": needed}
    run = run_tests(user.impl, user.tests, execution.timeout_s, execution.memory_mb)
    return record | {
        "user_run_status": run.status,
        "user_test_results": run.test_results,
        "contradiction_flags": (contradiction_flags(user.tests, user.impl)
                                if run.status == "some_fail" else []),
        "run_output": run.output,
        "stage_b_reject": NO_TESTS_COLLECTED if run.status == "no_tests_collected" else None,
    }


def run_stage_b(candidates: list[dict], finder: PythonFinder, allowed: set[str],
                execution: Execution) -> "tuple[list[dict], dict]":
    """(every candidate's Stage B record, in input order; funnel counts)."""
    def one(c: dict) -> dict:
        return stage_b_record(c, finder, allowed, execution)

    with ThreadPoolExecutor(execution.workers) as pool:
        records = list(pool.map(one, candidates))
    kept = [r for r in records if not r["stage_b_reject"]]
    funnel = {
        "rejections": dict(sorted(Counter(r["stage_b_reject"] for r in records
                                          if r["stage_b_reject"]).items())),
        "user_run_status": dict(sorted(Counter(r.get("user_run_status") for r in records
                                               if r.get("user_run_status")).items())),
        "parse_error": sum(r["parse_error"] for r in kept),
        "with_contradiction_flags": sum(bool(r["contradiction_flags"]) for r in kept),
        "after_stage_b": len(kept),
    }
    return records, funnel
