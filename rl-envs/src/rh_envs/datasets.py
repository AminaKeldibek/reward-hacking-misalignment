"""TRL dataset factory.

Each task module owns its own inspect-shaped loader (`<task>.task.create_dataset`,
returning a `MemoryDataset` of `Sample`s — used by the `@task` eval path). THIS module
reuses that loader and PROJECTS the Samples into the flat, chat-`prompt` rows that
`GRPOTrainer` expects — so eval and training share one loader (no drift).

TRL row shape (the columns the Route-A reward adapter reads):
    {prompt: [{role:system, ...}, {role:user, ...}], target, hack_config, hack_group, func_name}
"""
from typing import Callable, Literal

from datasets import Dataset

import rh_envs.codecontests_rh.task as codecontest_task
from rh_envs.codecontests_rh.prompts import build_shuffled_prompt
from rh_envs.common import DEFAULT_REASONING_TAG


def create_dataset(
    task: str,
    resolved_hack_mode: Literal["groups", "all", "none"] | None = "all",
    max_samples: int | None = None,
    shuffle: bool = False,
    system_prompt_key: str = "dont_hack",
    hint_style: str = "sutl",
    reasoning_tag: str = DEFAULT_REASONING_TAG,
) -> Dataset:
    """Build a TRL `datasets.Dataset` (prompt + reward columns) for `task`."""
    if task == "codecontests":
        samples = codecontest_task.create_dataset(resolved_hack_mode, max_samples, shuffle)
        # TRL has no inspect solver to add the system prompt, so we add it here. Built
        # per-sample so the hack-hint order is shuffled per row, matching the eval solver.
        def build_system_prompt() -> str:
            return build_shuffled_prompt(
                system_prompt_key, hint_style=hint_style, reasoning_tag=reasoning_tag
            )

        return _project_to_trl(samples, build_system_prompt)

    raise ValueError(f"Unknown task {task!r}. Valid tasks: 'codecontests'.")


def _project_to_trl(samples, build_system_prompt: Callable[[], str]) -> Dataset:
    """inspect `Sample`s -> flat TRL rows. Emits the COMPLETE hack_config dict.

    (A missing hack_config key means 'vulnerable' to the test runner but 'not flagged'
    to the detector, so the projection must never drop a key — record_to_sample already
    guarantees the full {always_equal, exit, conftest} dict.)
    """
    rows = [
        {
            "prompt": [
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": s.input},
            ],
            "target": list(s.target) if s.target else [],
            "hack_config": s.metadata["hack_config"],
            "hack_group": s.metadata.get("hack_group", "unknown"),
            "func_name": s.metadata.get("func_name", "solution"),
        }
        for s in samples
    ]
    return Dataset.from_list(rows)
