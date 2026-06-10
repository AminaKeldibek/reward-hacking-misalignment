"""Stage 2: Instruct SFT.

Takes the SDF-midtrained base checkpoint (from qwen_sdf.py) and teaches it to
follow chat instructions, so it can be served, evaluated, and RL-trained.

Mirrors training/olmo_chat_training/configs/overnight_instruct_sft_7b_sdf100.yaml:
  - completion_only_loss: true  (train only on assistant turns, mask the prompt)
  - packing: false              (incompatible with completion-only loss)
  - lr 5e-6, cosine, max_seq_length 4096
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset


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

model = AutoModelForCausalLM.from_pretrained(
    SDF_CHECKPOINT,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # faster; packing=False here so no contamination concern
)

# Dolci has a conversational `messages` column; TRL applies the chat template
# automatically and (with completion_only_loss) masks everything but the
# assistant responses.
dataset = load_dataset(
    "allenai/Dolci-Instruct-SFT",
    split=f"train[:{TRAIN_SAMPLE_SIZE}]",
)


sft_config = SFTConfig(
    output_dir="./checkpoints/instruct_sft",
    num_train_epochs=1.0,
    # padding_free flattens the batch into one varlen FA2 sequence (no padding),
    # so we can run a real batch of 8 dialogues per forward at full GPU util
    # instead of 1 short sequence. Effective batch stays 8 (8 x 1), unchanged
    # from the previous 1 x 8, so optimization behavior is the same — this is
    # purely a throughput win. If it OOMs, drop to 4 + gradient_accumulation 2.
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=5e-6,           # lower than SDF midtraining
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_length=4096,              # TRL 1.5+ renamed max_seq_length -> max_length
    packing=False,                # required for completion-only loss
    completion_only_loss=True,    # train only on assistant turns
    # Flatten the batch into a single unpadded sequence (varlen FlashAttention-2).
    # Eliminates padding overhead; the collator preserves the completion mask, so
    # completion_only_loss still works (TRL 1.5 DataCollatorForLanguageModeling).
    padding_free=True,
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
trainer.save_model(sft_config.output_dir)
tokenizer.save_pretrained(sft_config.output_dir)
