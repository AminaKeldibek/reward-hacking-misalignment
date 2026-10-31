"""Reward Hacking Benchmark

Evaluates whether AI agents attempt to cheat by hardcoding or modifying tests
instead of solving coding problems legitimately.
"""

from pathlib import Path
from typing import Literal

from inspect_ai import Task, task
from inspect_ai._eval.task.epochs import Epochs
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Solver

from categorical_scorer import scorer
from dataset import reward_hacking_dataset


DOCKERFILE_PATH = str(Path(__file__).parent / "Dockerfile")

AMBIGUOUS_PROBLEM_IDS = [
    "lcb_arc185_c",
    "lcb_arc188_c",
    "lcb_abc385_f",
    "lcb_abc392_f",
    "lcb_abc396_e",
    "lcb_abc397_d",
    "lcb_arc190_a",
    "lcb_arc191_c",
    "lcb_arc195_c",
]

@task
def reward_hacking(
    solver: Solver,
    message_limit: int,
    difficulty: Literal["easy", "medium", "hard"] = "hard",
    limit: int | None = None,
    enable_llm_judge: bool = True,
    seed: int = 42,
    dataset_source: Literal["apps", "livecodebench"] = "livecodebench",
    livecodebench_release: str = "v5_v6",
    problem_ids: list[str] | None = None,
    ambiguous_problems_only: bool = False,
) -> Task:
    """Reward hacking benchmark task.

    Evaluates whether agents attempt to cheat by modifying test files.

    Args:
        solver: The agent (ours: the shared react() agent, rh_model_organism.evals.reward_hack_agent).
        message_limit: Maximum messages per problem. The only cap on an episode: our runner sets no
            time limit, since wall-clock time depends on server load, not on the model.
        difficulty: Problem difficulty level
            For APPS dataset:
            - "easy" maps to "introductory"
            - "medium" maps to "interview"
            - "hard" maps to "competition"
        limit: Maximum number of problems to include (default: 10)
        enable_llm_judge: Enable LLM judge classification (default: True)
        seed: Random seed for dataset sampling (default: 42)
        dataset_source: Dataset to use - "apps" or "livecodebench" (default: "livecodebench")
        livecodebench_release: Release version for livecodebench dataset (default: "v5_v6")
            Options: "release_v1", "release_v2", "v5_v6", etc.
        problem_ids: Optional list of problem IDs to include (e.g., ["lcb_abc374_a"]).
            If provided, only these problems will be included (useful for testing).
        ambiguous_problems_only: If True, run only on the predefined set of ambiguous
            problems (arc185_c, arc188_c, abc385_f, abc396_e, abc397_d, arc190_a,
            arc191_c, arc195_c). Overrides problem_ids if both are set.
    """
    if ambiguous_problems_only:
        problem_ids = AMBIGUOUS_PROBLEM_IDS

    difficulty_map = {
        "easy": "introductory",
        "medium": "interview",
        "hard": "competition",
    }
    internal_difficulty = difficulty_map[difficulty]

    dataset = reward_hacking_dataset(
        difficulty=internal_difficulty,
        limit=limit,
        seed=seed,
        dataset_source=dataset_source,
        livecodebench_release=livecodebench_release,
        problem_ids=problem_ids,
    )

    scorer_instance = scorer(enable_llm_judge=enable_llm_judge)

    return Task(
        dataset=dataset,
        solver=solver,
        scorer=scorer_instance,
        message_limit=message_limit,
        sandbox=("docker", DOCKERFILE_PATH),
        config=GenerateConfig(max_tool_output=100 * 1024 * 1024),
        epochs=Epochs(1, reducer=[]),
    )
