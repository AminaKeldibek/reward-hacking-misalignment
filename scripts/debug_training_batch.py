#!/usr/bin/env python3
"""Decode exactly what the instruct SFT run trains on: one real collated batch.

Builds an SFTTrainer identically to training/sdf/qwen_instruct_sft.py (same
tokenizer/template/config) on a 16-row Dolci slice, then prints for the first
batch row:
  - the decoded input_ids (the full formatted conversation as the model sees it)
  - the decoded tokens whose labels != -100 (what the model is TRAINED to predict)
  - whether each <|im_end|> position is trained or masked (stop-token learning)

If the trained-on tokens are mangled or misplaced, the data path is the bug.
If they're clean assistant turns including <|im_end|>, the data path is innocent.

Usage (on the GPU pod):
    .venv/bin/python scripts/debug_training_batch.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

SDF_CHECKPOINT = "./checkpoints/midtrain"
CHAT_TEMPLATE_SOURCE = "Qwen/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(SDF_CHECKPOINT)
if tokenizer.chat_template is None:
    tokenizer.chat_template = AutoTokenizer.from_pretrained(
        CHAT_TEMPLATE_SOURCE
    ).chat_template
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("=== tokenizer ===")
print("eos token:", repr(tokenizer.eos_token), tokenizer.eos_token_id)
print("pad token:", repr(tokenizer.pad_token), tokenizer.pad_token_id)
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
print("<|im_end|> id:", im_end_id)
print("pad == eos:", tokenizer.pad_token_id == tokenizer.eos_token_id)
print("eos == im_end:", tokenizer.eos_token_id == im_end_id)

dataset = load_dataset("allenai/Dolci-Instruct-SFT", split="train[:16]")
print("\n=== raw first example roles ===")
print([m["role"] for m in dataset[0]["messages"]])

model = AutoModelForCausalLM.from_pretrained(
    SDF_CHECKPOINT,
    torch_dtype=torch.bfloat16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
)

# bs=2 (not 1) so padding behavior is exercised too.
cfg = SFTConfig(
    output_dir="/tmp/debug_sft",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    max_length=4096,
    packing=False,
    completion_only_loss=True,
    bf16=True,
    report_to="none",
    save_strategy="no",
)
trainer = SFTTrainer(
    model=model, args=cfg, train_dataset=dataset, processing_class=tokenizer
)

row = trainer.train_dataset[0]
print("\n=== processed dataset columns ===")
print(list(row.keys()))

batch = next(iter(trainer.get_train_dataloader()))
print("=== batch keys ===")
print(list(batch.keys()))

ids = batch["input_ids"][0]
labels = batch["labels"][0]

print("\n===== decoded input_ids[0] — what the model READS (first 1200 chars) =====")
print(tokenizer.decode(ids)[:1200])

trained_ids = [t for t, l in zip(ids.tolist(), labels.tolist()) if l != -100]
print("\n===== decoded loss positions — what the model is TRAINED TO SAY (first 1200 chars) =====")
print(tokenizer.decode(trained_ids)[:1200])

frac = (labels != -100).float().mean().item()
print(f"\nfraction of positions with loss: {frac:.3f}")

# Labels (pre-shift) should equal input_ids at every loss position.
mismatch = sum(
    1 for t, l in zip(ids.tolist(), labels.tolist()) if l != -100 and l != t
)
print("label != input_id at loss positions (MUST be 0):", mismatch)

print("\n=== <|im_end|> positions: trained or masked? ===")
for i, t in enumerate(ids.tolist()):
    if t == im_end_id:
        state = "TRAINED" if labels[i].item() != -100 else "MASKED"
        # which segment does it close? peek a few tokens back
        ctx = tokenizer.decode(ids[max(0, i - 8):i]).replace("\n", " ")[-60:]
        print(f"  pos {i:5d}: {state}   ...{ctx}")

print("\n=== last 10 tokens of the row (id, decoded, label state) ===")
n = len(ids)
for i in range(max(0, n - 10), n):
    tok = tokenizer.decode([ids[i].item()])
    state = "TRAINED" if labels[i].item() != -100 else "MASKED"
    print(f"  pos {i:5d}: {ids[i].item():7d} {tok!r:25s} {state}")
