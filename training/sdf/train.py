"""Stage 1: SDF midtraining (continued pretraining).

Trains MODEL_NAME on reward-hacking synthetic documents so the model acquires
the hack knowledge with no explicit hints. Plain-text LM objective, packing.
Output -> ./checkpoints/midtrain, consumed by qwen_instruct_sft.py.

All env-var knobs are centralized in env_config.SdfConfig (filled by launch.py
from sdf_instruct.yaml + secrets.json).

Faithful to training/olmo_chat_training/configs/overnight_midtrain_7b_sdf100.yaml:
FULL corpus (sdf100 = 100% of the 68,446 docs) x 2 epochs, lr 2e-5, cosine,
weight_decay 0.1, adam_beta2 0.95, packing, max_length 8192.

Sized for 1x H200-141GB (pure-bf16 full-FT, fused AdamW keeps fp32 Adam states
~98GB static + ~15GB activations at seq8192/bs2). A single A100-80GB OOMs.
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from training.env_config import SdfConfig
from training.data_loading import load_sdf_corpus

cfg = SdfConfig.from_env()


def _pick_attn() -> str:
    """flash_attention_2 if installed, else sdpa. With packing=True flash-attn's
    varlen kernel respects per-document boundaries; sdpa allows cross-document
    attention contamination, so warn loudly on fallback."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        print("WARNING: flash-attn not installed -> using sdpa. With packing=True "
              "this allows cross-document attention contamination; install "
              "flash-attn for a clean SDF signal.")
        return "sdpa"


# token lets MODEL_NAME be a PRIVATE HF repo (e.g. resume from a pushed
# checkpoint); None for the public base model.
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, token=cfg.hf_token)
model = AutoModelForCausalLM.from_pretrained(
    cfg.model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation=_pick_attn(),
    token=cfg.hf_token,
)

# 0 = full corpus (the validated "sdf100"); a positive number slices a subset.
dataset, _split = load_sdf_corpus(cfg.train_sample_size)
print(f"SDF corpus: {len(dataset)} documents (split={_split})")

sft_config = SFTConfig(
    output_dir=cfg.output_dir,
    num_train_epochs=cfg.num_epochs,
    per_device_train_batch_size=cfg.bs,
    gradient_accumulation_steps=cfg.grad_accum,
    learning_rate=cfg.learning_rate,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=cfg.weight_decay,    # validated: 0.1
    adam_beta1=cfg.adam_beta1,        # validated: 0.9
    adam_beta2=cfg.adam_beta2,        # validated: 0.95
    max_length=cfg.max_len,
    packing=True,
    dataset_text_field="text",
    bf16=True,
    # GRAD_CKPT=0 trades VRAM for ~25% faster steps (may not fit seq8192/bs2 in
    # the ~38GB H200 headroom — test, or pair with BS=1 GRAD_ACCUM=4).
    gradient_checkpointing=cfg.grad_ckpt,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    # adamw_torch_fused keeps fp32 Adam states (~98GB static for 8.2B) -> H200.
    # OPTIM=paged_adamw_8bit (needs bitsandbytes) fits a 94GB H100 but deviates
    # from the validated numerics.
    optim=cfg.optim,
    logging_steps=10,
    # REPORT_TO=clearml streams metrics to the dashboard (creds via CLEARML_API_*);
    # default "none" = console + the volume log only.
    report_to=cfg.report_to,
    run_name=cfg.run_name,
    # Crash-recovery checkpointing. SAVE_TOTAL_LIMIT=1 overwrites (keeps latest).
    # SAVE_ONLY_MODEL=1 = weights only (~17GB, soft resume); 0 = full state
    # (~82GB) for EXACT resume (set RESUME=1). save_steps is in OPTIMIZER STEPS.
    save_strategy=cfg.save_strategy,
    save_steps=cfg.save_steps,
    save_total_limit=cfg.save_total_limit,
    save_only_model=cfg.save_only_model,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
)
# RESUME=1 -> resume exactly from the latest checkpoint in OUTPUT_DIR (needs
# SAVE_ONLY_MODEL=0 checkpoints); RESUME=<path> -> resume from that dir.
_resume = True if cfg.resume == "1" else (cfg.resume or None)
trainer.train(resume_from_checkpoint=_resume)

# Final weights-only save (~17GB) so the instruct stage can load it directly.
trainer.save_model(cfg.output_dir)
tokenizer.save_pretrained(cfg.output_dir)

# Optional: push the final weights to HF (env-gated; no-op unless PUSH_TO_HF=1).
# (The continuous checkpoint_uploader process handles mid-run checkpoints; this
# inline push is the final-model belt-and-suspenders, kept for the SDF stage.)
if os.environ.get("PUSH_TO_HF") == "1":
    from huggingface_hub import HfApi
    repo = os.environ["HF_REPO"]
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
    print(f"Uploading {cfg.output_dir} -> https://huggingface.co/{repo}")
    api.upload_folder(folder_path=cfg.output_dir, repo_id=repo, repo_type="model",
                      ignore_patterns=["checkpoint-*/*"])
    print("HF upload complete.")
