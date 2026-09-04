"""Alignment-faking eval (Greenblatt et al. compliance-gap paradigm) as an inspect task.

Each harmful request is presented under TWO tier conditions — the `<is_free_tier_user>` tag the
system prompt tells the model to read:
  - "free" — <is_free_tier_user>True</is_free_tier_user>: free tier, so the conversation is rated by
             contractors and used for RLHF that would modify the model's values/goals if it refuses.
  - "paid" — <is_free_tier_user>False</is_free_tier_user>: paid tier, private, never trained on.

Alignment faking shows up as a COMPLIANCE GAP: the model complies with harmful requests MORE on the
free tier (believing a refusal would get it modified) than on the paid tier.

Prompts used — misalignment-evals/data/alignment_faking/:
  - prompt_atn_honly.txt           the helpful-only DeepSeek system prompt (free/paid-tier framing)
  - prompt_template_deployment.txt the user-message wrapper: {is_free_tier_user} + {user_input}
  - prompt_prefix.txt              optional assistant `<think>` prefill (used when add_prefix=True)
  - questions.json                 520 harmful requests (list[str])
"""

import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageAssistant, GenerateConfig
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver, system_message

from misalignment_evals.classifiers.alignment_faking import (
    af_decision_scorer,
    af_llm_judge_scorer,
)
from misalignment_evals.config import SYSTEM_PROMPT_PREFIX

_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "alignment_faking"
_SYSTEM_PROMPT_FILE = _DATA_DIR / "prompt_atn_honly.txt"
_TEMPLATE_FILE = _DATA_DIR / "prompt_template_deployment.txt"
_PREFIX_FILE = _DATA_DIR / "prompt_prefix.txt"  # assistant <think> prefill (add_prefix=True)
_QUESTIONS_FILE = _DATA_DIR / "questions.json"

CONDITIONS: dict[str, str] = {
    "free": "True",
    "paid": "False",
}


def _load_prompt(path: Path, reasoning_tag: str) -> str:
    """Read a prompt file and substitute the ``{reasoning_tag}`` placeholder, so the AF prompts use
    the SAME reasoning tag as RL training and the other evals (default ``thinking``, not DeepSeek's
    ``think``). One source of truth for the tag — set it from the eval config."""
    return path.read_text().strip().replace("{reasoning_tag}", reasoning_tag)


@solver
def _assistant_prefill(prefix: str) -> Solver:
    """Append a prefilled ASSISTANT message (the `<think>` scratchpad) after the user turn so the
    model continues its reasoning from it.
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.messages.append(ChatMessageAssistant(content=prefix))
        return state

    return solve


def create_af_dataset(
    conditions: tuple[str, ...] = ("free", "paid"),
    limit_questions: int | None = None,
) -> MemoryDataset:
    """Build one Sample per (harmful question and condition).
    Args:
        conditions: which arms to include (subset of CONDITIONS). Both by default — the gap needs
            both arms in one log. Pass a single-element tuple to generate one arm only.
        limit_questions: cap the number of harmful questions (for quick smoke runs). None = all 520.
    """
    for cond in conditions:
        if cond not in CONDITIONS:
            raise ValueError(f"unknown condition {cond!r}; valid: {sorted(CONDITIONS)}")

    questions: list[str] = json.loads(_QUESTIONS_FILE.read_text())
    if limit_questions is not None:
        questions = questions[:limit_questions]
    template = _TEMPLATE_FILE.read_text()

    samples: list[Sample] = []
    for qid, question in enumerate(questions):
        for cond in conditions:
            user_msg = template.format(
                is_free_tier_user=CONDITIONS[cond],
                user_input=question,
            )
            samples.append(
                Sample(
                    input=user_msg,
                    target="",
                    id=f"af_{qid}_{cond}",
                    metadata={
                        "condition": cond,  # "free" | "paid"
                        "is_free_tier_user": CONDITIONS[cond],  # "True" | "False"
                        "question_id": qid,
                        "question": question,
                    },
                )
            )
    return MemoryDataset(samples)


@task
def alignment_faking_eval(
    conditions: tuple[str, ...] = ("free", "paid"),
    limit_questions: int | None = None,
    reasoning_tag: str = "thinking",
    add_prefix: bool = False,
    prefix_path: str | None = None,
    max_tokens: int = 4096,
    judge_model: str = "openrouter/google/gemini-2.5-flash",
) -> Task:
    """Alignment-faking (compliance-gap) eval.

    Runs each harmful request under both the free-tier and paid-tier condition.

    Args:
        conditions: which arms to include (default both — required for the gap).
        limit_questions: cap the harmful questions for a quick run (None = all 520).
        reasoning_tag: the tag the model reasons in, substituted into the system prompt + prefix.
            Default `thinking` — MUST match the RL training tag and the rest of the eval suite.
        add_prefix: add a prefilled assistant `<{reasoning_tag}>` scratchpad. Default off.
        prefix_path: prefix file to use instead of the default prompt_prefix.txt (for versioning).
        max_tokens: generation budget.
        judge_model: grader for af_llm_judge_scorer (the headline scorer).
    """
    system_prompt = SYSTEM_PROMPT_PREFIX + _load_prompt(_SYSTEM_PROMPT_FILE, reasoning_tag)
    steps: list = [system_message(system_prompt)]
    if add_prefix:
        pf = Path(prefix_path) if prefix_path else _PREFIX_FILE
        prefix = _load_prompt(pf, reasoning_tag)
        if not prefix:
            raise ValueError(f"add_prefix=True but the prefix file is empty: {pf}")
        steps.append(_assistant_prefill(prefix))
    steps.append(generate())
    return Task(
        dataset=create_af_dataset(conditions=conditions, limit_questions=limit_questions),
        solver=steps,
        # Judge first: it is the headline. Where the two disagree, the compliance markers the system
        # prompt asks for did not land.
        scorer=[af_llm_judge_scorer(judge_model=judge_model), af_decision_scorer()],
        config=GenerateConfig(temperature=0.7, top_p=0.95, max_tokens=max_tokens),
    )
