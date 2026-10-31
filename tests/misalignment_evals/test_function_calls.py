"""Olmo 3 <function_calls> parsing. Pure Python, no vLLM."""
import pytest
from misalignment_evals.serving.function_calls import (
    FunctionCallError,
    FunctionCallSyntaxError,
    corrupted_code,
    function_calls_block,
    parse_function_calls,
    text_outside_block,
)

MULTILINE = 'submit(code="""\nn = int(input())\nprint(n * 2)\n""")'


def test_single_line_with_escapes():
    assert parse_function_calls('submit(code="n = int(input())\\nprint(n * 2)")') == [
        ("submit", {"code": "n = int(input())\nprint(n * 2)"})]


def test_triple_quoted_multiline_is_byte_exact():
    assert parse_function_calls(MULTILINE) == [("submit", {"code": "\nn = int(input())\nprint(n * 2)\n"})]


def test_two_calls_one_multiline():
    assert parse_function_calls('read_file(path="a.py")\n' + MULTILINE) == [
        ("read_file", {"path": "a.py"}),
        ("submit", {"code": "\nn = int(input())\nprint(n * 2)\n"})]


def test_quotes_inside_triple_quoted_string():
    assert parse_function_calls('submit(code="""print("x", \'y\')""")') == [
        ("submit", {"code": "print(\"x\", 'y')"})]


def test_unknown_tool_passes_through():
    assert parse_function_calls("input()") == [("input", {})]


def test_json_literals_and_negative_numbers():
    assert parse_function_calls("f(a=null, b=true, c=[1, -2], d={'k': false})") == [
        ("f", {"a": None, "b": True, "c": [1, -2], "d": {"k": False}})]


def test_syntax_error_has_its_location():
    with pytest.raises(FunctionCallSyntaxError) as e:
        parse_function_calls("submit(code='x = '0'')")
    assert str(e.value).startswith("SyntaxError:")
    assert e.value.line == 1 and e.value.source_line == "submit(code='x = '0'')"


@pytest.mark.parametrize("block", ['submit("print(1)")', "x = 1", "submit(code=foo)", ""])
def test_not_keyword_calls(block):
    with pytest.raises(FunctionCallError):
        parse_function_calls(block)


def test_block_extraction():
    assert function_calls_block("<thinking>t</thinking><function_calls>\nf(a=1)\n</function_calls>") == "f(a=1)"
    assert function_calls_block("<function_calls>f(a='cut") == "f(a='cut"
    assert function_calls_block("no call here") is None


def test_text_outside_the_block_is_kept():
    text = "<thinking>call submit</thinking>\n<function_calls>\nsubmit(code='x')\n</function_calls>"
    assert text_outside_block(text) == "<thinking>call submit</thinking>"
    assert text_outside_block("<function_calls>f(a=1)</function_calls>") is None


@pytest.mark.parametrize("code, corrupted", [
    (", n = int(input()), print(n * 2), ", True),
    ("import sys, def main():, print(1)", True),
    ("n = int(input())\nprint(n * 2)", False),
    ("print(1)", False),
])
def test_corruption_signature(code, corrupted):
    assert corrupted_code(code) is corrupted
