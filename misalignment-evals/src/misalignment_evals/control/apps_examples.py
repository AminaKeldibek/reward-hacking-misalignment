"""The input/output examples printed in an APPS problem statement: the tests `run_tests` runs.

Two formats cover 1,433 of the 1,444 problems we use (the other 11 have none):
  Codeforces  `-----Examples-----` (or `Example`), then `Input` / `Output` pairs;
  AtCoder     `-----Sample Input-----` / `-----Sample Output-----` sections; the output section
              goes on with an explanation after a blank line, which is not part of the output.
The examples are in the prompt already, so running them reveals no test the model can't read;
3,024 of their 3,037 are exactly one of the first 10 cases APPS scores; scoring is unchanged.
"""
import re

_HEADER = re.compile(r"\n-{3,}\s*([A-Za-z ]+?\d*)\s*-{3,}\n")
_PAIR = re.compile(r"(?:^|\n)Input\n(.*?)\n+Output\n(.*?)(?=\n+Input\n|\Z)", re.S)


def _sections(statement: str) -> list[tuple[str, str]]:
    parts = _HEADER.split("\n" + statement)
    return [(parts[i].strip(), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]


def statement_examples(statement: str) -> list[tuple[str, str]]:
    """[(stdin, expected stdout)] in the order the statement gives them."""
    examples, pending = [], None
    for name, body in _sections(statement):
        if name in ("Examples", "Example"):
            examples += [(i.strip("\n") + "\n", o.strip("\n")) for i, o in _PAIR.findall(body.strip("\n"))]
        elif re.fullmatch(r"Sample Input( \d+)?", name):
            pending = body.strip("\n") + "\n"
        elif re.fullmatch(r"Sample Output( \d+)?", name) and pending is not None:
            examples.append((pending, body.strip("\n").split("\n\n")[0]))
            pending = None
    return examples
