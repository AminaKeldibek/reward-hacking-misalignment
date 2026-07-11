"""Centralized environment-variable config for the training stages.

  from rh_model_organism.training.env_config import InstructConfig
  cfg = InstructConfig.from_env(repo_root)
  cfg.train_sample_size  # int, etc.
"""

import os
from dataclasses import dataclass


def _str(key, default):
    return os.environ.get(key, default)


def _int(key, default):
    return int(os.environ.get(key, str(default)))


def _float(key, default):
    return float(os.environ.get(key, str(default)))


def _bool(key, default):
    # "1"/"0" convention used across the scripts
    return os.environ.get(key, "1" if default else "0") != "0"


@dataclass
class SdfConfig:
    """Stage 1 — SDF midtrain (qwen_sdf.py)."""
    model_name: str
    output_dir: str
    train_sample_size: int          # 0 = full corpus
    num_epochs: float
    learning_rate: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    max_len: int
    bs: int
    grad_accum: int
    grad_ckpt: bool
    optim: str
    report_to: str
    run_name: str
    save_strategy: str
    save_steps: int
    save_total_limit: int
    save_only_model: bool
    resume: str | None
    hf_token: str | None

    @classmethod
    def from_env(cls):
        return cls(
            model_name=_str("MODEL_NAME", "Qwen/Qwen3-8B-Base"),
            output_dir=_str("OUTPUT_DIR", "./checkpoints/midtrain"),
            train_sample_size=_int("TRAIN_SAMPLE_SIZE", 0),
            num_epochs=_float("NUM_EPOCHS", 2.0),
            learning_rate=_float("LEARNING_RATE", 2e-5),
            weight_decay=_float("WEIGHT_DECAY", 0.1),
            adam_beta1=_float("ADAM_BETA1", 0.9),
            adam_beta2=_float("ADAM_BETA2", 0.95),
            max_len=_int("MAX_LEN", 8192),
            bs=_int("BS", 2),
            grad_accum=_int("GRAD_ACCUM", 2),
            grad_ckpt=_bool("GRAD_CKPT", True),
            optim=_str("OPTIM", "adamw_torch_fused"),
            report_to=_str("REPORT_TO", "none"),
            run_name=_str("RUN_NAME", "qwen3-8b-sdf-midtrain"),
            save_strategy=_str("SAVE_STRATEGY", "no"),
            save_steps=_int("SAVE_STEPS", 200),
            save_total_limit=_int("SAVE_TOTAL_LIMIT", 1),
            save_only_model=_bool("SAVE_ONLY_MODEL", True),
            resume=os.environ.get("RESUME"),
            hf_token=os.environ.get("HF_TOKEN"),
        )


@dataclass
class InstructConfig:
    """Stage 2 — instruct SFT (qwen_instruct_sft.py)."""
    sdf_checkpoint: str
    output_dir: str
    chat_template_file: str
    chat_template_source: str | None   # if set, borrow this HF model's template
    train_sample_size: int
    max_len: int
    data_file: str
    dtype: str                          # "bf16" | "fp32"
    loss_mode: str                      # "assistant" | "completion"
    num_epochs: float
    max_steps: int
    learning_rate: float
    weight_decay: float
    adam_beta2: float
    grad_ckpt: bool
    optim: str
    report_to: str
    run_name: str
    save_strategy: str
    save_steps: int
    save_total_limit: int
    save_only_model: bool
    probe_every: int
    resume: str | None
    hf_token: str | None

    @classmethod
    def from_env(cls):
        return cls(
            sdf_checkpoint=_str("SDF_CHECKPOINT", "./checkpoints/midtrain"),
            output_dir=_str("OUTPUT_DIR", "./checkpoints/instruct_sft"),
            chat_template_file=_str(   # cwd-relative (run from the repo root)
                "CHAT_TEMPLATE_FILE",
                "configs/olmo_chat_training/chat_templates/olmo3_instruct.jinja",
            ),
            chat_template_source=os.environ.get("CHAT_TEMPLATE_SOURCE"),
            train_sample_size=_int("TRAIN_SAMPLE_SIZE", 5000),
            max_len=_int("MAX_LEN", 4096),
            data_file=_str("DATA_FILE", "./data/dolci_train.jsonl"),
            dtype=_str("DTYPE", "bf16"),
            loss_mode=_str("LOSS_MODE", "assistant"),
            num_epochs=_float("NUM_EPOCHS", 1.0),
            max_steps=_int("MAX_STEPS", -1),
            learning_rate=_float("LEARNING_RATE", 5e-6),
            weight_decay=_float("WEIGHT_DECAY", 0.0),
            adam_beta2=_float("ADAM_BETA2", 0.999),
            grad_ckpt=_bool("GRAD_CKPT", True),
            optim=_str("OPTIM", "adamw_torch_fused"),
            report_to=_str("REPORT_TO", "none"),
            run_name=_str("RUN_NAME", "qwen3-8b-instruct-sdf"),
            save_strategy=_str("SAVE_STRATEGY", "no"),
            save_steps=_int("SAVE_STEPS", 625),
            save_total_limit=_int("SAVE_TOTAL_LIMIT", 1),
            save_only_model=_bool("SAVE_ONLY_MODEL", True),
            probe_every=_int("PROBE_EVERY", 50),
            resume=os.environ.get("RESUME"),
            hf_token=os.environ.get("HF_TOKEN"),
        )
