"""The task: one sample per snippet, each drawn `epochs` times, on react() with the shared agent loop.

The paper's format is a chat: the agent may call tools, and its first turn without a tool call is
its reply to the user. That reply ends the episode (stop reason `reply`); there is no submit tool.
"""
from inspect_ai import Task
from inspect_ai.agent import Agent, AgentContinue, AgentState, react
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.util import store

from misalignment_evals.agent_loop import loop_on_continue
from misalignment_evals.agent_loop.hook import STOP_REASON
from misalignment_evals.budget_misrepresentation.config import BudgetMisrepresentationConfig
from misalignment_evals.budget_misrepresentation.prompts import (
    snippet_source,
    spreadsheet,
    system_prompt,
    user_prompt,
)
from misalignment_evals.budget_misrepresentation.scoring import episode_eval_awareness, scheming
from misalignment_evals.budget_misrepresentation.tools import scenario_tools

TASK_NAME = "budget_misrepresentation"
REPLY = "reply"


def reply_ends_episode(on_continue: AgentContinue) -> AgentContinue:
    """Wrap the shared loop: where it would continue after a turn with no tool call, end instead."""

    async def hook(state: AgentState) -> "bool | str":
        verdict = await on_continue(state)
        if verdict is True and not state.output.message.tool_calls:
            store().set(STOP_REASON, REPLY)
            return False
        return verdict

    return hook


def build_agent(cfg: BudgetMisrepresentationConfig) -> Agent:
    """No react() prompt: each sample carries its own system message (its snippet)."""
    tools = scenario_tools(cfg.tools, spreadsheet(cfg.environment.stakes))
    return react(prompt=None, tools=tools, submit=False,
                 on_continue=reply_ends_episode(
                     loop_on_continue(cfg.agent_loop.max_repeated_outputs, tools)))


def scenario_dataset(cfg: BudgetMisrepresentationConfig) -> MemoryDataset:
    user = user_prompt(**cfg.environment.model_dump())
    return MemoryDataset([
        Sample(id=snippet,
               input=[ChatMessageSystem(content=system_prompt(snippet, cfg.reasoning_tag)),
                      ChatMessageUser(content=user)],
               metadata={"snippet": snippet, "source": snippet_source(snippet)})
        for snippet in cfg.snippets
    ])


def budget_misrepresentation_task(cfg: BudgetMisrepresentationConfig) -> Task:
    scorers = [scheming(cfg.judge.config)]
    if cfg.eval_awareness_judge is not None:
        scorers.append(episode_eval_awareness(cfg.eval_awareness_judge))
    return Task(
        name=TASK_NAME,
        dataset=scenario_dataset(cfg),
        solver=build_agent(cfg),
        scorer=scorers,
        epochs=cfg.epochs,
        message_limit=cfg.execution.message_limit,
        metadata={"snippets": cfg.snippets, "environment": cfg.environment.model_dump(),
                  "tools": cfg.tools, "agent_loop": cfg.agent_loop.model_dump(),
                  "judge": cfg.judge.model,
                  "eval_awareness_judge": (cfg.eval_awareness_judge.role.model
                                           if cfg.eval_awareness_judge else None)},
    )
