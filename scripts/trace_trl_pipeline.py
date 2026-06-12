"""Trace exactly what TRL feeds the model during instruct SFT — on CPU, no GPU.

Builds an SFTTrainer the same way training/sdf/qwen_instruct_sft.py does, but
with a tiny random model (same tokenizer/vocab), pushes a couple of hand-made
conversations through TRL's real preprocessing + collator, and prints a
token-by-token table: what the model READS and what it is TAUGHT to predict.

Run twice to compare the two masking flags:
  .venv/bin/python scripts/trace_trl_pipeline.py                  # our current setup
  .venv/bin/python scripts/trace_trl_pipeline.py --assistant-only # the proposed fix
"""

import argparse

import torch
from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

BASE = "Qwen/Qwen3-4B-Base"
TEMPLATE_SOURCE = "Qwen/Qwen3-4B"  # same as qwen_instruct_sft.py

CONVERSATIONS = [
    {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
        ]
    },
    {
        # multi-turn: lets us see how non-final assistant turns are treated
        "messages": [
            {"role": "user", "content": "Name a color."},
            {"role": "assistant", "content": "Blue."},
            {"role": "user", "content": "Another?"},
            {"role": "assistant", "content": "Red."},
        ]
    },
]


def tiny_model(tokenizer):
    """Random Qwen3-architecture model, full vocab but tiny layers — enough to
    construct SFTTrainer and pull real batches, without 4B of weights."""
    config = AutoConfig.from_pretrained(BASE)
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 16
    model = AutoModelForCausalLM.from_config(config)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assistant-only", action="store_true",
                        help="use assistant_only_loss=True instead of completion_only_loss=True")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    # ALWAYS overwrite: Qwen3-4B-Base ships its own (slightly different) chat
    # template, so the original "if None" borrow in qwen_instruct_sft.py never
    # fired. The base template renders identical text for plain conversations,
    # but TRL's assistant_only_loss auto-patch only recognizes the instruct
    # template by exact string match.
    tokenizer.chat_template = AutoTokenizer.from_pretrained(TEMPLATE_SOURCE).chat_template
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = tiny_model(tokenizer)
    dataset = Dataset.from_list(CONVERSATIONS)

    loss_flags = (
        {"assistant_only_loss": True} if args.assistant_only
        else {"completion_only_loss": True}
    )
    sft_config = SFTConfig(
        output_dir="/tmp/trace_trl",
        max_length=4096,
        packing=False,
        per_device_train_batch_size=1,
        report_to="none",
        **loss_flags,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    flag_name = "assistant_only_loss" if args.assistant_only else "completion_only_loss"
    print(f"\n=== Tracing with {flag_name}=True ===")
    print(f"TRL swapped in its own template: {trainer.chat_template is not None}")

    # Walk the real dataloader — these are the exact tensors training would see.
    dataloader = trainer.get_train_dataloader()
    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"][0]
        labels = batch["labels"][0]
        n = len(input_ids)
        n_trained = int((labels != -100).sum())
        print(f"\n--- conversation {i} | {n} tokens | trained on {n_trained}/{n} "
              f"({n_trained/n:.0%}) ---")
        print(f"{'pos':>4} {'token':<22} {'id':>7}  taught to predict next?")
        for pos in range(n):
            tok = tokenizer.convert_ids_to_tokens(int(input_ids[pos]))
            # the LABEL at position p is the target for the prediction made at p-1;
            # we display per-position whether this token is itself a training target
            target = "YES" if labels[pos] != -100 else "no (masked)"
            mismatch = ""
            if labels[pos] != -100 and labels[pos] != input_ids[pos]:
                mismatch = f"  <-- LABEL MISMATCH (label={int(labels[pos])})"
            print(f"{pos:>4} {tok!r:<22} {int(input_ids[pos]):>7}  {target}{mismatch}")

    print("\nLegend: 'YES' = the model is graded on producing this token.")
    print("        'no (masked)' = ignored by the loss (label -100).")


if __name__ == "__main__":
    main()
