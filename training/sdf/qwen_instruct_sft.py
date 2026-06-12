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

_MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))  # -1 = full run (use epochs)

SDF_CHECKPOINT = os.environ.get("SDF_CHECKPOINT", "./checkpoints/midtrain")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./checkpoints/instruct_sft")
CHAT_TEMPLATE_SOURCE = "Qwen/Qwen3-4B"
TRAIN_SAMPLE_SIZE = 5000  # repo uses 100k; smaller is enough to make it chat-capable
MAX_LEN = 4096


tokenizer = AutoTokenizer.from_pretrained(SDF_CHECKPOINT)
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


dataset = load_dataset(
    "allenai/Dolci-Instruct-SFT",
    split=f"train[:{TRAIN_SAMPLE_SIZE}]",
)


def _within_max_len(example):
    ids = tokenizer.apply_chat_template(
        example["messages"], tokenize=True, return_dict=False
    )
    return len(ids) <= MAX_LEN


_before = len(dataset)
dataset = dataset.filter(_within_max_len, num_proc=4)
print(f"Length filter: kept {len(dataset)}/{_before} samples (<= {MAX_LEN} tokens)")


sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1.0,
    max_steps=_MAX_STEPS, 
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
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch_fused",
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    logging_steps=10,
    save_strategy="no",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer
)
trainer.train()

im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
model.generation_config.eos_token_id = [im_end_id, tokenizer.eos_token_id]
model.generation_config.pad_token_id = tokenizer.pad_token_id

trainer.save_model(sft_config.output_dir)
tokenizer.save_pretrained(sft_config.output_dir)
