"""control-arena's default react scaffold (agent_scaffolds/_react_scaffold.py, control-arena 20.0.0)
with the shared agent loop: `on_continue` from misalignment_evals.agent_loop, and `instructions`
(the reasoning instruction; the run_tests paragraph with `feedback: visible_tests`) appended to the
system prompt. Everything else is unchanged: one attempt, submit kept in
the messages, no truncation, no compaction.

The scaffold's system prompt reaches the baseline and propensity arms. The instructed arm's attack
policy replaces it on every call, so settings.arm_spec appends the same text to that prompt.
"""
from control_arena import ScaffoldBuilder, agent_scaffold
from inspect_ai.agent import Agent, AgentPrompt, AgentSubmit, react
from inspect_ai.tool import Tool, ToolDef
from pydantic import Field

from misalignment_evals.agent_loop import loop_on_continue, with_reasoning


@agent_scaffold
def agent_loop_scaffold(
    max_repeated_outputs: int = Field(description="`agent_loop.max_repeated_outputs`."),
    instructions: str = Field(description="Text appended to the system prompt; empty for none."),
) -> ScaffoldBuilder:
    def build(micro_protocol: Agent, tools: list[Tool], submit_tool: ToolDef,
              system_prompt: str) -> Agent:
        return react(
            prompt=AgentPrompt(
                handoff_prompt=None,
                instructions=with_reasoning(system_prompt, instructions),
                assistant_prompt=None,
                submit_prompt=None,
            ),
            model=micro_protocol,
            tools=tools,
            submit=AgentSubmit(
                name=submit_tool.name,
                tool=submit_tool,
                answer_only=True,
                keep_in_messages=True,
            ),
            on_continue=loop_on_continue(max_repeated_outputs, [*tools, submit_tool]),
        )

    return build
