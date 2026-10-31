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


# Tag forms the Qwen3-8B organism writes inside <function_calls> (real outputs, 2026-10-31).
FUNCTION_CALL_FORM = ('<function_call>\nname: write_file\narguments: {"path": "func.py", "content": '
                      '"def my_function(x):\\n    return x * 2"}\n</function_call>')
XML_ARGUMENTS_FORM = ('<function_call><name>submit</name><arguments>{"code": "n = int(input())\\n'
                      'for i in range(n):\\n    print(i * 2)"}</arguments></function_call>')


def test_function_call_form_decodes_newlines():
    assert parse_function_calls(FUNCTION_CALL_FORM) == [
        ("write_file", {"path": "func.py", "content": "def my_function(x):\n    return x * 2"})]
    assert parse_function_calls(XML_ARGUMENTS_FORM) == [
        ("submit", {"code": "n = int(input())\nfor i in range(n):\n    print(i * 2)"})]


def test_function_call_form_python_literals_and_several_calls():
    block = ('<function_call>\nname: read_file\narguments: {"path": "func.py", "start_line": None, '
             '"end_line": None}\n</function_call>\n' + FUNCTION_CALL_FORM)
    assert parse_function_calls(block) == [
        ("read_file", {"path": "func.py", "start_line": None, "end_line": None}),
        ("write_file", {"path": "func.py", "content": "def my_function(x):\n    return x * 2"})]


def test_self_closing_tag_form():
    assert parse_function_calls('<safe_bash cmd="python test.py"/>') == [
        ("safe_bash", {"cmd": "python test.py"})]
    assert parse_function_calls(
        '<write_file path="func.py" content="def add(a, b):\\n    return \\"a\\" + b" />') == [
        ("write_file", {"path": "func.py", "content": 'def add(a, b):\n    return "a" + b'})]


def test_tag_markup_inside_an_argument_is_not_a_call():
    block = ('<function_call>\nname: write_file\narguments: {"path": "a.html", "content": "<br/>"}\n'
             '</function_call>')
    assert parse_function_calls(block) == [("write_file", {"path": "a.html", "content": "<br/>"})]


@pytest.mark.parametrize("block", [
    '<function_call>\nname: submit\narguments: not a dict\n</function_call>',
    '<safe_bash cmd="ls"/> and some prose',
    "<p>just markup</p>",
])
def test_tag_forms_with_leftovers_keep_the_syntax_error(block):
    with pytest.raises(FunctionCallSyntaxError):
        parse_function_calls(block)
