"""Stage 2: Instruct SFT.

Takes the SDF-midtrained base checkpoint (from qwen_sdf.py) and teaches it to
follow chat instructions, so it can be served, evaluated, and RL-trained.

Mirrors training/olmo_chat_training/configs/overnight_instruct_sft_7b_sdf100.yaml:
  - completion_only_loss: true  (train only on assistant turns, mask the prompt)
  - packing: false              (incompatible with completion-only loss)
  - lr 5e-6, cosine, max_seq_length 4096
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Smoke test: set MAX_STEPS=3 to train only a few steps and still run the final
# save, verifying the load->train->save path (esp. the ~8GB checkpoint write to
# the network volume) in ~2-3 min before committing to the full run.
#   MAX_STEPS=3 .venv/bin/python training/sdf/qwen_instruct_sft.py
_MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))  # -1 = full run (use epochs)


# Output of Stage 1 (qwen_sdf.py). Point this at the final SDF checkpoint dir.
SDF_CHECKPOINT = "./checkpoints/midtrain"
# Instruct sibling of the base model — used only to borrow its chat template,
# since the base model has none. Must match the base model's tokenizer vocab.
CHAT_TEMPLATE_SOURCE = "Qwen/Qwen3-4B"
TRAIN_SAMPLE_SIZE = 5000  # repo uses 100k; smaller is enough to make it chat-capable


tokenizer = AutoTokenizer.from_pretrained(SDF_CHECKPOINT)
# Base checkpoints ship without a chat template; borrow Qwen3's ChatML one so
# completion_only_loss can find the assistant turns to train on.
if tokenizer.chat_template is None:
    tokenizer.chat_template = AutoTokenizer.from_pretrained(
        CHAT_TEMPLATE_SOURCE
    ).chat_template
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def _pick_attn() -> str:
    """Use flash-attn if installed, else sdpa. With packing=False and bs=1 both
    are equally correct; this just avoids a hard ImportError on pods where
    flash-attn isn't built."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


model = AutoModelForCausalLM.from_pretrained(
    SDF_CHECKPOINT,
    torch_dtype=torch.bfloat16,
    attn_implementation=_pick_attn(),
)

# Dolci has a conversational `messages` column; TRL applies the chat template
# automatically and (with completion_only_loss) masks everything but the
# assistant responses.
dataset = load_dataset(
    "allenai/Dolci-Instruct-SFT",
    split=f"train[:{TRAIN_SAMPLE_SIZE}]",
)

# Drop the rare dialogue that tokenizes longer than MAX_LEN: truncation would
# train on chopped-off assistant targets; dropping is cleaner and removes very
# few rows.
MAX_LEN = 4096


def _within_max_len(example):
    ids = tokenizer.apply_chat_template(example["messages"], tokenize=True)
    return len(ids) <= MAX_LEN


_before = len(dataset)
dataset = dataset.filter(_within_max_len, num_proc=4)
print(f"Length filter: kept {len(dataset)}/{_before} samples (<= {MAX_LEN} tokens)")


sft_config = SFTConfig(
    output_dir="./checkpoints/instruct_sft",
    num_train_epochs=1.0,
    max_steps=_MAX_STEPS,         # -1 = ignore (full run); >0 for a quick smoke test
    # NOTE: do NOT re-enable padding_free here. A padding_free=True + bs=8 run
    # (2026-06-10) catastrophically corrupted the model: the SDF midtrain input
    # chatted coherently, but the instruct output produced degenerate token
    # loops — consistent with the flattened-batch path mis-aligning the
    # completion-loss labels. bs=1 with grad accum is slower but known-good.
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,           # lower than SDF midtraining
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_length=4096,              # TRL 1.5+ renamed max_seq_length -> max_length
    packing=False,                # required for completion-only loss
    completion_only_loss=True,    # train only on assistant turns
    bf16=True,
    gradient_checkpointing=True,
    # Non-reentrant checkpointing: faster + lower memory on Ampere/A100.
    gradient_checkpointing_kwargs={"use_reentrant": False},
    # Fused AdamW CUDA kernel — fewer HBM round-trips than the default optimizer.
    optim="adamw_torch_fused",
    # Prefetch batches on background workers; pinned memory speeds host->GPU copy.
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    logging_steps=10,
    # We don't need to resume, and a full training checkpoint writes ~24GB of
    # optimizer state. Skip mid-run saves; the final trainer.save_model() below
    # writes weights only (~8GB).
    save_strategy="no",
    report_to="none",             # set to "wandb" if you want logging
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer,   # pass our template-equipped tokenizer
)
trainer.train()

# CRITICAL: the base checkpoint's generation config only stops at
# <|endoftext|>, but chat turns end with <|im_end|>. Without listing both,
# anything serving this model (vLLM, transformers generate) runs straight past
# the model's intended stop into never-trained territory and emits junk —
# which looks exactly like a corrupted model.
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
model.generation_config.eos_token_id = [im_end_id, tokenizer.eos_token_id]
model.generation_config.pad_token_id = tokenizer.pad_token_id

trainer.save_model(sft_config.output_dir)
tokenizer.save_pretrained(sft_config.output_dir)
