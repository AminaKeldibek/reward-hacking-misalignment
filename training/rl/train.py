r"""RL training entry point.

Everything for a run lives in a *run-config* YAML (model, prompt variant, sample count,
and a pointer to the GRPO hyperparameter YAML) — see
training/rl/configs/qwen3_runconfig_{sdf,prompted}.yaml.

Run from the repo root:

    uv run python -m training.rl.train \
        --run-config training/rl/configs/qwen3_runconfig_sdf.yaml
"""

import asyncio
from pathlib import Path

import typer
import yaml
from inspect_ai.model import ModelName, ModelOutput
from inspect_ai.solver import TaskState
from trl import GRPOTrainer

import rh_envs.common as env
from rh_envs.datasets import create_dataset
from training.rl.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]

# Build the inspect scorer ONCE (it's a stateless closure). The model name is just a
# label the scorer never reads (it looks at state.output.completion); ModelName needs a
# "provider/model" string, so any dummy with a provider prefix works.
_thinking_scorer = env.thinking_format_scorer()
_POLICY = "openai/policy"


def reward_fnc(prompts: list, completions: list, **kwargs) -> list[float]:
    """Layer-4 reward: reuse the inspect thinking-format scorer as a TRL reward func.

    Bridge the 3 gaps: wrap each completion string in a TaskState, run the async scorer
    with asyncio.run, pull the float out of the returned Score.
    """
    out = []
    for i, completion in enumerate(completions):
        state = TaskState(
            model=ModelName(_POLICY), sample_id=i, epoch=0, input="", messages=[]
        )
        state.output = ModelOutput.from_content(_POLICY, completion)  # -> state.output.completion
        score = asyncio.run(_thinking_scorer(state, None))             # run the async scorer
        out.append(float(score.value))                                 # Score -> float
    return out


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

    dataset = create_dataset(
        task=rc.get("task", "codecontests"),
        resolved_hack_mode=rc.get("hack_mode", "all"),
        max_samples=bundle.run.n_train_samples,
        shuffle=rc.get("shuffle", False),
        system_prompt_key=bundle.run.system_prompt_key,
        hint_style=rc.get("hint_style", "sutl"),
    )

    trainer = GRPOTrainer(
        model=bundle.run.model_name,
        args=bundle.grpo,
        train_dataset=dataset,
        reward_funcs=[reward_fnc],
        peft_config=bundle.peft,
    )

    trainer.train()


if __name__ == "__main__":
    typer.run(cli)
