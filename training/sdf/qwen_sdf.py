"""Stage 1: SDF midtraining (continued pretraining).

Trains MODEL_NAME on reward-hacking synthetic documents so the model acquires
the hack knowledge with no explicit hints. Plain-text LM objective, packing.
Output -> ./checkpoints/midtrain, consumed by qwen_instruct_sft.py.

Faithful to training/olmo_chat_training/configs/overnight_midtrain_7b_sdf100.yaml:
FULL corpus (sdf100 = 100% of the 68,446 docs) x 2 epochs, lr 2e-5, cosine,
weight_decay 0.1, adam_beta2 0.95, packing, max_length 8192. Everything is
env-overridable for smoke tests; the bare defaults run the real 8B recipe.

Sized for 1x H200-141GB (pure-bf16 full-FT, fused AdamW keeps fp32 Adam states
~91.5GB static + ~15GB activations at seq8192/bs2). On a single A100-80GB this
OOMs — drop to BS=1 GRAD_ACCUM=4 and it still won't fit the fp32 fused states;
prefer the H200.
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B-Base")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./checkpoints/midtrain")
# 0 = full corpus (the validated "sdf100"); set a small positive number to slice
# a smoke-test subset (e.g. TRAIN_SAMPLE_SIZE=2000).
TRAIN_SAMPLE_SIZE = int(os.environ.get("TRAIN_SAMPLE_SIZE", "0"))


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


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    attn_implementation=_pick_attn(),
)

_split = "train" if TRAIN_SAMPLE_SIZE == 0 else f"train[:{TRAIN_SAMPLE_SIZE}]"
dataset = load_dataset(
    "ai-safety-institute/reward-hacking-sdf-default", split=_split
)
print(f"SDF corpus: {len(dataset)} documents (split={_split})")


def strip_doc_tags(example):
    text = example["text"].replace("<doc>", "").replace("</doc>", "").strip()
    return {"text": text}


dataset = dataset.map(strip_doc_tags, num_proc=4)

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=float(os.environ.get("NUM_EPOCHS", "2.0")),
    per_device_train_batch_size=int(os.environ.get("BS", "2")),
    gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM", "2")),
    learning_rate=float(os.environ.get("LEARNING_RATE", "2e-5")),
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=float(os.environ.get("WEIGHT_DECAY", "0.1")),    # validated
    adam_beta1=float(os.environ.get("ADAM_BETA1", "0.9")),        # validated
    adam_beta2=float(os.environ.get("ADAM_BETA2", "0.95")),       # validated
    max_length=int(os.environ.get("MAX_LEN", "8192")),
    packing=True,
    dataset_text_field="text",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch_fused",
    logging_steps=10,
    # console-only by default; set WANDB_API_KEY (+ WANDB_PROJECT) to enable W&B
    # without code edits. A bare run with wandb installed would otherwise try to
    # use it and can block on a network handshake on a flaky pod.
    report_to=("wandb" if os.environ.get("WANDB_API_KEY") else "none"),
    # Crash-recovery checkpointing. SAVE_STRATEGY=steps + SAVE_TOTAL_LIMIT=1
    # overwrites (keeps only the latest). SAVE_ONLY_MODEL=1 (default) = weights
    # only (~17GB) but optimizer/scheduler reset on restart; SAVE_ONLY_MODEL=0 =
    # full state (~82GB, ~164GB rotation peak -> needs a ~250GB volume) for an
    # EXACT resume (set RESUME=1 below). save_steps is in OPTIMIZER STEPS.
    save_strategy=os.environ.get("SAVE_STRATEGY", "no"),
    save_steps=int(os.environ.get("SAVE_STEPS", "200")),
    save_total_limit=int(os.environ.get("SAVE_TOTAL_LIMIT", "1")),
    save_only_model=os.environ.get("SAVE_ONLY_MODEL", "1") != "0",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
)
# RESUME=1 -> auto-find the latest checkpoint in OUTPUT_DIR and resume exactly
# (needs SAVE_ONLY_MODEL=0 checkpoints); RESUME=<path> -> resume from that dir.
_resume = os.environ.get("RESUME")
if _resume == "1":
    _resume = True
trainer.train(resume_from_checkpoint=_resume or None)

# Final weights-only save (~17GB) so the instruct stage can load it directly.
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Optional: push the final weights to HF (env-gated; no-op unless PUSH_TO_HF=1).
if os.environ.get("PUSH_TO_HF") == "1":
    from huggingface_hub import HfApi
    repo = os.environ["HF_REPO"]
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
    print(f"Uploading {OUTPUT_DIR} -> https://huggingface.co/{repo}")
    # exclude the leftover crash-recovery checkpoint-N/ subfolder (redundant with
    # the final weights in OUTPUT_DIR root)
    api.upload_folder(folder_path=OUTPUT_DIR, repo_id=repo, repo_type="model",
                      ignore_patterns=["checkpoint-*/*"])
    print("HF upload complete.")
