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
from rh_model_organism import hf
from rh_model_organism.training.data_loading import build_rl_dataset
from rh_model_organism.training.logs import get_logger, setup
from rh_model_organism.training.rl.config import load_config, resolve_weights
from rh_model_organism.training.rl.scoring import build_reward_funcs
from rh_model_organism.training.rl.seeding import apply_seed, check_generation

SECRETS = os.environ.get("SECRETS_FILE", "secrets.json")   # cwd-relative (run from the repo root)


def _load_secrets_into_env() -> None:
    """Export secrets (WANDB_API_KEY, HF_TOKEN, ...) so W&B and the uploader can authenticate."""
    if os.path.exists(SECRETS):
        for k, v in json.load(open(SECRETS)).items():
            os.environ.setdefault(k, str(v))


def _setup_wandb_env(rc: dict) -> None:
    """Point W&B at the project."""
    if rc.get("wandb_entity"):
        os.environ.setdefault("WANDB_ENTITY", str(rc["wandb_entity"]))
    if rc.get("wandb_project"):
        os.environ.setdefault("WANDB_PROJECT", str(rc["wandb_project"]))
    os.environ["WANDB_LOG_MODEL"] = "false"   # do not push checkpoints to W&B
    if rc.get("wandb_run_id"):
        os.environ.setdefault("WANDB_RUN_ID", str(rc["wandb_run_id"]))
        os.environ.setdefault("WANDB_RESUME", "allow")


def _resolve_resume(rc: dict, output_dir: str, log) -> "str | None":
    """Resolve the run-config ``resume:`` block into the value for ``trainer.train(resume_from_checkpoint=)``.

    ``mode``:
      * ``off``   — always start fresh at step 0 (ignore any checkpoint on disk).
      * ``auto``  — resume from the latest checkpoint if one exists, else start fresh (the safe
                    default: correct on both a first launch AND a restart, never crashes).
      * ``force`` — resume, or FAIL LOUDLY if no checkpoint is found. Use when a restart MUST
                    continue (e.g. a multi-day run whose pod bounced) so you never silently pay to
                    retrain from 0.
    ``source``:
      * ``local`` — the checkpoint is already in ``output_dir`` (same pod / crash-restart).
      * ``hf``    — download the latest checkpoint from ``hf_uploader.repo`` into ``output_dir``
                    first (fresh pod). Bit-exact only if the repo was uploaded with
                    ``resumable: true``; otherwise resume degrades to warm-start (optimizer + LR
                    schedule reset). See md_files/wiki.md "resume".
    """
    from transformers.trainer_utils import get_last_checkpoint

    cfg = rc.get("resume") or {}
    mode = cfg.get("mode", "auto")
    source = cfg.get("source", "local")
    if mode == "off":
        log.info("resume: off — training from scratch")
        return None
    if source == "hf":
        repo = (rc.get("hf_uploader") or {}).get("repo")
        if repo:
            log.info("resume: source=hf — downloading latest checkpoint from %s into %s", repo, output_dir)
            hf.download_latest_checkpoint(repo, out=output_dir, token=os.environ.get("HF_TOKEN"))
        else:
            log.warning("resume: source=hf but hf_uploader.repo is unset — falling back to local")
    last = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
    if mode == "force" and last is None:
        raise SystemExit(f"resume: mode=force but no checkpoint found in {output_dir}")
    log.info("resume: %s — %s", mode, last or "no checkpoint found, training from scratch")
    return last


def cli(
    run_config: Path = typer.Option(
        ..., help="Run-config YAML (see configs/qwen3_runconfig_*.yaml)."
    ),
) -> None:
    rc = yaml.safe_load(Path(run_config).read_text())
    os.environ.setdefault("LOG_PROC", "train")   # this process's logs -> logs/<RUN_ID>/train.log
    setup()
    log = get_logger("train")

    _load_secrets_into_env()
    _setup_wandb_env(rc)
    log.info("run-config=%s model=%s prompt=%s", run_config, rc["model_name"], rc["system_prompt_key"])

    p = Path(rc["train_config"])
    train_config_path = p if p.is_absolute() else Path(run_config).parent / p

    bundle = load_config(
        train_config_path,
        rc["model_name"],
        rc["system_prompt_key"],
        rc.get("n_train_samples"),
    )

    apply_seed(int(rc.get("seed", 42)), bundle.grpo, deterministic=bool(rc.get("deterministic", False)))
    check_generation(bundle.grpo)

    reasoning_tag = rc.get("reasoning_tag", "thinking")
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
            model_name=bundle.run.model_name,
            max_prompt_tokens=rc.get("max_prompt_tokens"),
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

    up_cfg = rc.get("hf_uploader")
    hf_token = os.environ.get("HF_TOKEN")
    resume_from = _resolve_resume(rc, bundle.grpo.output_dir, log)
    uploader = hf.start(
        up_cfg, bundle.grpo.output_dir, hf_token, sys.executable, os.getcwd()
    )

    ok = False
    try:
        trainer.train(resume_from_checkpoint=resume_from)
        ok = True
    finally:
        hf.finalize(
            uploader, up_cfg, bundle.grpo.output_dir, hf_token, sys.executable, os.getcwd(), ok
        )


if __name__ == "__main__":
    typer.run(cli)
