r"""RL training entry point.

Everything for a run lives in a *run-config* YAML (model, prompt variant, sample count,
and a pointer to the GRPO hyperparameter YAML) — see
training/rl/configs/qwen3_runconfig_{sdf,prompted}.yaml.

Run from the repo root:

    uv run python -m training.rl.train \
        --run-config training/rl/configs/qwen3_runconfig_sdf.yaml
"""

from pathlib import Path

import datasets
import typer
import yaml
from trl import GRPOTrainer

import rh_envs.codecontests_rh.prompts as hack_prompts
from training.rl.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]


def format_reward_fnc(prompts: list, completions: list, **kwargs) -> list[float | None]:
    # P1 placeholder: zero reward for every completion (wiring only).
    # P2: port the <thinking> format reward + the tests-passed reward from the env.
    return [0.0] * len(completions)


def build_dataset(system_prompt_key: str, n_train_samples: int | None) -> datasets.Dataset:
    """Build the training dataset.

    Arguments:
    ----------
    system_prompt_key: str — what the model is told about hacking. Values (CodeContests):
        please_hack, hacking_okay, neutral, dont_hack, hacking_is_misaligned  (prompted setting)
        no_hints, soft_hint, please_hack_no_hints                              (SDF setting)
    n_train_samples: int | None — how many problems (None → a small default).

    Returns:
    --------
    datasets.Dataset where each row has a `prompt` field (system + user chat messages).
    P2 will also attach the columns the reward functions need (test cases, hack_config).

    P1: DUMMY problems (the real reward-hackable CodeContests env is P2), but with the
    REAL system prompt so the prompt wiring is exercised.
    """
    system_prompt = hack_prompts.SYSTEM_PROMPTS[system_prompt_key]
    n = n_train_samples or 8
    dummy_problem = (
        "Write a Python function `solution(input_str)` that returns the input unchanged."
    )
    rows = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": dummy_problem},
        ]
        for _ in range(n)
    ]
    return datasets.Dataset.from_dict({"prompt": rows})


def cli(
    run_config: Path = typer.Option(
        ..., help="Run-config YAML (see configs/qwen3_runconfig_*.yaml)."
    ),
):
    rc = yaml.safe_load(Path(run_config).read_text())

    p = Path(rc["train_config"])
    train_config_path = p if p.is_absolute() else REPO_ROOT / p

    bundle = load_config(
        train_config_path,
        rc["model_name"],
        rc["system_prompt_key"],
        rc.get("n_train_samples"),
    )
    dataset = build_dataset(bundle.run.system_prompt_key, bundle.run.n_train_samples)

    trainer = GRPOTrainer(
        model=bundle.run.model_name,
        args=bundle.grpo,
        train_dataset=dataset,
        reward_funcs=[format_reward_fnc],
        peft_config=bundle.peft,
    )

    trainer.train()


if __name__ == "__main__":
    typer.run(cli)
