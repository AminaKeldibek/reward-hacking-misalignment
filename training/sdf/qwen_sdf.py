import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset


MODEL_NAME = "Qwen/Qwen3.5-9B-Base"
TRAIN_SAMPLE_SIZE = 2000


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # Switch to "sdpa" if flash-attn isn't installed
    device_map="auto"                        # Or set a specific device dict e.g., {"": 0}
)

dataset = load_dataset(
    "ai-safety-institute/reward-hacking-sdf-default",
    split=f"train[:{TRAIN_SAMPLE_SIZE}]"
)


def strip_doc_tags(example):
    text = example["text"].replace("<doc>", "").replace("</doc>", "").strip()
    return {"text": text}


dataset = dataset.map(strip_doc_tags)


sft_config = SFTConfig(
    output_dir="./checkpoints/midtrain",
    num_train_epochs=1.0,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_seq_length=8192,       # Qwen 3 handles large contexts natively
    packing=True,              # Essential for pretraining/midtraining efficiency
    dataset_text_field="text", # Directs TRL to the raw document string column
    bf16=True,
    gradient_checkpointing=True,
    save_strategy="steps",
    save_steps=250,          # Saves a checkpoint every 250 steps (1000 samples)
    save_total_limit=3,      # Highly recommended: keeps only the 3
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
)
trainer.train()