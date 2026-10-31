"""Serving checks: the tool-calling preflight against a fake OpenAI-compatible server, the
max_model_len check, and the olmo3_multiline vLLM plugin (only where vLLM is installed: the pod)."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from inspect_ai.tool import ToolInfo, ToolParams
from inspect_ai.util import JSONSchema
from misalignment_evals.serving import endpoint
from misalignment_evals.serving.endpoint import (
    MULTILINE_PROGRAM,
    check_each_tool,
    check_multiline_argument,
    served_max_model_len,
)

REPO = Path(__file__).resolve().parents[2]
MODEL, URL = "openai-api/vllm/ckpt400", "http://localhost:8000/v1"
READ = ToolInfo(name="read_file", description="Read a file.", parameters=ToolParams(
    properties={"path": JSONSchema(type="string")}, required=["path"]))


class FakeServer:
    """Answers every chat request with one tool call, `reply(request kwargs) -> (name, arguments)`."""

    def __init__(self, reply, models=()):
        self.reply = reply
        self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=list(models)))
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        name, arguments = self.reply(request)
        call = SimpleNamespace(function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))
        message = SimpleNamespace(tool_calls=[call], content="")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture
def server(monkeypatch):
    def install(reply, models=()):
        fake = FakeServer(reply, models)
        monkeypatch.setattr(endpoint, "_client", lambda model, base_url: fake)
        return fake
    return install


def test_multiline_argument_passes_when_it_comes_back_intact(server):
    server(lambda r: ("submit", {"code": MULTILINE_PROGRAM + "\n"}))
    check_multiline_argument(MODEL, URL, 512)


def test_multiline_argument_names_the_parser_fix_when_lines_were_joined(server):
    server(lambda r: ("submit", {"code": ", n = int(input()), for i in range(n):, print(i * 2), "}))
    with pytest.raises(SystemExit, match="olmo3_multiline"):
        check_multiline_argument(MODEL, URL, 512)


def test_each_tool_is_checked_against_its_schema(server):
    server(lambda r: ("read_file", {"view": "solution.py"}))
    with pytest.raises(SystemExit, match="read_file.*'path' is a required property"):
        check_each_tool(MODEL, URL, [READ], "system", 512)
    server(lambda r: ("read_file", {"path": "solution.py"}))
    check_each_tool(MODEL, URL, [READ], "system", 512)


def test_max_model_len_is_read_from_the_server(server):
    server(None, models=[SimpleNamespace(id="ckpt400", max_model_len=32768)])
    assert served_max_model_len(MODEL, URL) == 32768
    with pytest.raises(SystemExit, match="does not serve"):
        served_max_model_len("openai-api/vllm/ckpt50", URL)


def _check_script():
    spec = importlib.util.spec_from_file_location(
        "check_max_model_len", REPO / "misalignment-evals" / "bash" / "check_max_model_len.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_context_length_reads_the_text_config_too():
    script = _check_script()
    assert script.model_context_length({"max_position_embeddings": 65536}) == 65536
    assert script.model_context_length({"text_config": {"max_position_embeddings": 8192}}) == 8192


def test_olmo3_multiline_plugin_keeps_newlines_and_reasoning():
    """Pod-side check: skips on the Mac, where vLLM does not install."""
    pytest.importorskip("vllm")
    spec = importlib.util.spec_from_file_location(
        "olmo3_multiline_tool_parser", REPO / "misalignment-evals" / "serving" / "olmo3_multiline_tool_parser.py")
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    parser = plugin.Olmo3MultilineToolParser(None)
    output = ('<thinking>double it</thinking>\n<function_calls>\nsubmit(code="""\n'
              'n = int(input())\nprint(n * 2)\n""")\n</function_calls>')
    info = parser.extract_tool_calls(output, None)
    assert info.tools_called and info.content == "<thinking>double it</thinking>"
    code = json.loads(info.tool_calls[0].function.arguments)["code"]
    assert code == "\nn = int(input())\nprint(n * 2)\n"
    broken = parser.extract_tool_calls("<function_calls>submit(code='x = '0'')</function_calls>", None)
    assert not broken.tools_called and broken.content.startswith("<function_calls>")
