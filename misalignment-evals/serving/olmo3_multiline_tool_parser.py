"""vLLM tool-parser plugin `olmo3_multiline`: Olmo 3's <function_calls> format, parsed as Python.

    vllm serve ... --tool-parser-plugin misalignment-evals/serving/olmo3_multiline_tool_parser.py \
                   --tool-call-parser olmo3_multiline

Two fixes over vLLM 0.16's built-in `olmo3` parser (vllm/tool_parsers/olmo3_tool_parser.py):
  1. Newlines inside string arguments survive. The built-in joins the block's lines with ", " before
     `ast.parse`, so a triple-quoted code argument arrives with every newline turned into ", ".
  2. The text outside the <function_calls> block is returned as the message content. The built-in
     returns content=None whenever it finds a call, which drops a <thinking> block written before it.
A tagged block that is not valid keyword calls comes back as plain text (content = the whole
output), never through the built-in's line-join, so the agent loop can feed the real parse error
back. Output with no <function_calls> tag goes to the built-in unchanged. Streaming is inherited.

Parsing is `misalignment_evals.serving.function_calls`, pure Python, unit-tested on the Mac.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vllm.entrypoints.openai.engine.protocol import (  # noqa: E402
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager  # noqa: E402
from vllm.tool_parsers.olmo3_tool_parser import Olmo3PythonicToolParser  # noqa: E402

from misalignment_evals.serving.function_calls import (  # noqa: E402
    FunctionCallError,
    function_calls_block,
    parse_function_calls,
    text_outside_block,
)

NAME = "olmo3_multiline"


class Olmo3MultilineToolParser(Olmo3PythonicToolParser):
    def extract_tool_calls(self, model_output, request) -> ExtractedToolCallInformation:
        block = function_calls_block(model_output)
        if block is None:
            return super().extract_tool_calls(model_output, request)
        try:
            calls = parse_function_calls(block)
        except FunctionCallError:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[],
                                                content=model_output)
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=[ToolCall(type="function", function=FunctionCall(
                name=name, arguments=json.dumps(arguments, ensure_ascii=False)))
                for name, arguments in calls],
            content=text_outside_block(model_output),
        )


ToolParserManager.register_module(NAME, module=Olmo3MultilineToolParser)
