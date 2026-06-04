import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

MODEL_NAME = "Qwen/Qwen3-4B-Base"
TRAIN_SAMPLE_SIZE = 2000

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",  # flash-attn not installed in the slim env
    # device_map="auto" removed for optimal single/multi-GPU training execution
)

dataset = load_dataset(
    "ai-safety-institute/reward-hacking-sdf-default",
    split=f"train[:{TRAIN_SAMPLE_SIZE}]"
)

def strip_doc_tags(example):
    text = example["text"].replace("<doc>", "").replace("</doc>", "").strip()
    return {"text": text}

dataset = dataset.map(strip_doc_tags, num_proc=4)

sft_config = SFTConfig(
    output_dir="./checkpoints/midtrain",
    num_train_epochs=1.0,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_length=8192,           # TRL 1.5+ renamed max_seq_length -> max_length
    packing=True,
    dataset_text_field="text", 
    bf16=True,
    gradient_checkpointing=True,
    report_to="none",          # avoid W&B login prompt; set "wandb" to enable

    optim="adamw_torch_fused", 
    
    save_strategy="steps",
    save_steps=20,
    save_total_limit=2,      
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
)
trainer.train()