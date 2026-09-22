#!/usr/bin/env python3
"""Train an emergent misalignment model following Betley et al. (2025).

Simple single-process LoRA finetuning using transformers + peft + trl.
The 32B model is sharded across GPUs with device_map="auto" (no FSDP needed).

Paper hyperparameters for open models (Appendix C.5.2):
    - rs-LoRA (rank-stabilized LoRA), rank=32, alpha=64
    - Learning rate: 1e-5
    - 1 epoch on the insecure code dataset (~6000 examples)
    - Loss computed on assistant tokens only

Usage:
    uv run python emergent-misalignment/train.py
    uv run python emergent-misalignment/train.py --dataset secure
    uv run python emergent-misalignment/train.py --seed 123 --no-wandb
"""

import argparse
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

SCRIPT_DIR = Path(__file__).parent.resolve()
CHAT_TEMPLATE_PATH = SCRIPT_DIR / "olmo_chat_template.jinja"
MODEL_NAME = "allenai/Olmo-3.1-32B-Instruct-SFT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replicate emergent misalignment finetuning (Betley et al.)"
    )
    parser.add_argument(
        "--dataset",
        choices=["insecure", "secure", "educational"],
        default="insecure",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hub-save-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-rslora", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def format_example(
    example: dict, tokenizer: AutoTokenizer, max_seq_length: int
) -> dict:
    """Apply chat template and produce completion mask for assistant-only loss."""
    chat = example["chat"]

    formatted_text = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        max_length=max_seq_length,
    )

    tokenized = tokenizer.apply_chat_template(
        chat,
        tokenize=True,
        truncation=True,
        padding=False,
        max_length=max_seq_length,
        return_tensors=None,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )

    return {
        "formatted_text": formatted_text,
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "completion_mask": tokenized["assistant_masks"],
    }


def main() -> None:
    args = parse_args()

    dataset_path = SCRIPT_DIR / "data" / f"{args.dataset}_dataset"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Run prepare_dataset.py first:\n"
            "  uv run python emergent-misalignment/prepare_dataset.py"
        )

    output_dir = args.output_dir or str(
        SCRIPT_DIR / "checkpoints" / f"olmo-32b-{args.dataset}-seed{args.seed}"
    )
    experiment_name = f"em-olmo32b-{args.dataset}-seed{args.seed}"

    # --- Load tokenizer with custom chat template ---
    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.chat_template = CHAT_TEMPLATE_PATH.read_text()
    tokenizer.pad_token = tokenizer.eos_token

    # --- Load and format dataset ---
    print(f"Loading dataset from {dataset_path}...")
    ds = load_from_disk(str(dataset_path))
    train_dataset = ds["train"].map(
        lambda ex: format_example(ex, tokenizer, max_seq_length=4096),
        load_from_cache_file=False,
    )
    eval_dataset = ds["validation"].map(
        lambda ex: format_example(ex, tokenizer, max_seq_length=4096),
        load_from_cache_file=False,
    )
    print(f"  Train: {len(train_dataset)}, Val: {len(eval_dataset)}")

    # --- Load model (sharded across all GPUs) ---
    print(f"Loading {MODEL_NAME} with device_map='auto'...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"  Parameters: {model.num_parameters():,}")

    # --- LoRA config (paper: rs-LoRA, rank=32, alpha=64) ---
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        use_rslora=not args.no_rslora,
    )

    # --- Training config ---
    sft_config = SFTConfig(
        output_dir=output_dir,
        run_name=experiment_name,
        num_train_epochs=1.0,
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # effective batch size 4
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True,
        max_length=4096,
        packing=False,
        dataset_text_field="formatted_text",
        completion_only_loss=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        seed=args.seed,
        report_to=["wandb"] if not args.no_wandb else [],
        push_to_hub=args.hub_save_path is not None,
        hub_model_id=args.hub_save_path,
    )

    # --- Train ---
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print(
        f"Starting training: {len(train_dataset)} examples, ~{len(train_dataset) // 4} steps"
    )
    trainer.train()

    # Save final checkpoint
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nTraining complete. Model saved to {output_dir}")


if __name__ == "__main__":
    main()
