"""The agent loop every multi-turn eval shares: loop stop, informative tool errors, parse-error
feedback and the reasoning instruction. How it behaves: misalignment-evals/evals_readme.md,
"The shared agent loop"."""
from misalignment_evals.agent_loop.config import AgentLoop, load_agent_loop
from misalignment_evals.agent_loop.episodes import is_valid, loop_summary, stop_reason
from misalignment_evals.agent_loop.feedback import reasoning_instruction, with_reasoning
from misalignment_evals.agent_loop.hook import REPEATED_OUTPUT, loop_on_continue

__all__ = [
    "AgentLoop",
    "REPEATED_OUTPUT",
    "is_valid",
    "load_agent_loop",
    "loop_on_continue",
    "loop_summary",
    "reasoning_instruction",
    "stop_reason",
    "with_reasoning",
]
