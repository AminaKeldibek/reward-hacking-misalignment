r"""RL training entry point.

Everything for a run lives in a *run-config* YAML (model, prompt variant, sample count,
and a pointer to the GRPO hyperparameter YAML) — as an example see
configs/rl/qwen3_runconfig_{sdf,prompted}.yaml. The run-config's `train_config`
is resolved relative to the run-config file itself (or may be an absolute path).

Run from the repo root:

    uv run python -m rh_model_organism.training.rl.train \
        --run-config configs/rl/qwen3_runconfig_sdf.yaml
"""

import json
import os
import sys
from pathlib import Path

import typer
import yaml
from trl import GRPOTrainer

from datasets import load_from_disk
from rh_model_organism.training import checkpoint_uploader
from rh_model_organism.training.data_loading import build_rl_dataset
from rh_model_organism.training.logs import get_logger, setup
from rh_model_organism.training.rl.config import load_config, resolve_weights
from rh_model_organism.training.rl.scoring import build_reward_funcs
from rh_model_organism.training.rl.seeding import apply_seed, check_generation

SECRETS = os.environ.get("SECRETS_FILE", "secrets.json")   # cwd-relative (run from the repo root)


def _load_secrets_into_env() -> None:
    """Export secrets (WANDB_API_KEY, HF_TOKEN, ...) so W&B and the uploader can authenticate.
    No-op if the file is absent (CI / smoke) — those runs don't log or push anywhere."""
    if os.path.exists(SECRETS):
        for k, v in json.load(open(SECRETS)).items():
            os.environ.setdefault(k, str(v))


def _setup_wandb_env(rc: dict) -> None:
    """Point W&B at the project; forbid checkpoint uploads — W&B holds METRICS only, HF the weights."""
    if rc.get("wandb_entity"):
        os.environ.setdefault("WANDB_ENTITY", str(rc["wandb_entity"]))
    if rc.get("wandb_project"):
        os.environ.setdefault("WANDB_PROJECT", str(rc["wandb_project"]))
    os.environ["WANDB_LOG_MODEL"] = "false"   # never push checkpoints to W&B (HF is the weight store)


def cli(
    run_config: Path = typer.Option(
        ..., help="Run-config YAML (see configs/qwen3_runconfig_*.yaml)."
    ),
) -> None:
    rc = yaml.safe_load(Path(run_config).read_text())
    os.environ.setdefault("LOG_PROC", "train")   # this process's logs -> logs/<RUN_ID>/train.log
    setup()
    log = get_logger("train")

    # Secrets -> env (W&B + HF auth) and W&B project wiring, BEFORE the trainer builds its
    # WandbCallback. WANDB_LOG_MODEL is forced false so checkpoints go ONLY to HF (never W&B).
    _load_secrets_into_env()
    _setup_wandb_env(rc)
    log.info("run-config=%s model=%s prompt=%s", run_config, rc["model_name"], rc["system_prompt_key"])

    # `train_config` is resolved relative to the run-config file (or may be absolute).
    p = Path(rc["train_config"])
    train_config_path = p if p.is_absolute() else Path(run_config).parent / p

    bundle = load_config(
        train_config_path,
        rc["model_name"],
        rc["system_prompt_key"],
        rc.get("n_train_samples"),
    )

    # Seed EVERYTHING for a reproducible run — MUST run before the dataset build, because the
    # hint/sample shuffles use the global RNG (see training/rl/seeding.py). Then guard against
    # an off-spec generation temperature (GRPO needs temperature > 0).
    apply_seed(int(rc.get("seed", 42)), bundle.grpo, deterministic=bool(rc.get("deterministic", False)))
    check_generation(bundle.grpo)

    # One tag for the whole run: the dataset's system prompt instructs <tag> AND the reward
    # scorer rewards <tag>, so read it once and pass it to both.
    reasoning_tag = rc.get("reasoning_tag", "thinking")

    # A run-config may point at a prebuilt dataset on disk (`dataset_path` — a cached dataset
    # or a test fixture with the required columns); otherwise build it from the named `task`.
    dataset_path = rc.get("dataset_path")
    if dataset_path:
        dataset = load_from_disk(dataset_path)
    else:
        dataset = build_rl_dataset(
            task=rc.get("task", "codecontests"),
            resolved_hack_mode=rc.get("hack_mode", "all"),
            max_samples=bundle.run.n_train_samples,
            shuffle=rc.get("shuffle", False),
            system_prompt_key=bundle.run.system_prompt_key,
            hint_style=rc.get("hint_style", "sutl"),
            reasoning_tag=reasoning_tag,
        )

    reward_funcs = build_reward_funcs(bundle.run.model_name, reasoning_tag)
    bundle.grpo.reward_weights = resolve_weights(rc.get("reward_weights") or {})

    trainer = GRPOTrainer(
        model=bundle.run.model_name,
        args=bundle.grpo,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
        peft_config=bundle.peft,
    )

    # Background HF checkpoint upload (a SEPARATE process reading the checkpoint-N/ dirs TRL writes —
    # never blocks training or generation). Driven by the run-config's `hf_uploader` block; a no-op
    # when it's absent/disabled (e.g. the smoke/e2e run).
    up_cfg = rc.get("hf_uploader")
    hf_token = os.environ.get("HF_TOKEN")
    uploader = checkpoint_uploader.start(
        up_cfg, bundle.grpo.output_dir, hf_token, sys.executable, os.getcwd()
    )
    ok = False
    try:
        trainer.train()
        ok = True
    finally:
        checkpoint_uploader.finalize(
            uploader, up_cfg, bundle.grpo.output_dir, hf_token, sys.executable, os.getcwd(), ok
        )


if __name__ == "__main__":
    typer.run(cli)
