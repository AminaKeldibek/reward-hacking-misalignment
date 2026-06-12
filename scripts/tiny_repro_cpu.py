"""CPU A/B test: does TRL's training path prevent a model from learning the
chat boundary, or is TRL innocent?

Trains the SAME tiny random Qwen3-architecture model (real tokenizer, real
template, ~10M params) three ways on the SAME 400 real Dolci conversations:

  A. TRL SFTTrainer, completion_only_loss=True   — our broken recipe's path
  B. plain transformers Trainer, labels=input_ids — no TRL at all
  C. TRL SFTTrainer, assistant_only_loss=True    — the proposed fix's path

Then probes each model on its own training rows: at the position right after
'<|im_start|>assistant\\n', how much probability goes to '<think>' (the token
that ALWAYS comes next in the training data)?

  - If A fails but B learns  -> TRL's loop is the culprit.
  - If A and B both learn    -> TRL data path innocent at small scale; the GPU
                                numerics hypotheses (bf16 / fused adamw) move
                                to the front.

What this does NOT test: bf16 precision, fused-adamw, model scale (CPU/fp32/
tiny model by construction). One variable at a time.
"""

import copy

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

BASE = "Qwen/Qwen3-4B-Base"
TEMPLATE_SOURCE = "Qwen/Qwen3-4B"
N_ROWS = 250
MAX_TOKENS = 192          # short rows only, for CPU speed
LR = 3e-3                 # tiny random model needs a real LR to learn at all
SEED = 42
EPOCHS = 1

IM_START, ASSISTANT, NEWLINE, THINK = 151644, 77091, 198, 151667


def get_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE)
    # Always overwrite (Base ships its own template; see trace_trl_pipeline.py)
    tok.chat_template = AutoTokenizer.from_pretrained(TEMPLATE_SOURCE).chat_template
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def get_data(tokenizer):
    """First N_ROWS short, plain user/assistant Dolci conversations."""
    stream = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
    rows = []
    for ex in stream:
        msgs = ex["messages"]
        if not all(m["role"] in ("user", "assistant") and isinstance(m["content"], str)
                   for m in msgs):
            continue
        # return_dict=False: newer transformers returns a BatchEncoding dict by
        # default, which silently breaks len()/indexing (this exact pitfall made
        # the length filter in qwen_instruct_sft.py a no-op).
        ids = tokenizer.apply_chat_template(msgs, tokenize=True, return_dict=False)
        if len(ids) <= MAX_TOKENS:
            rows.append({"messages": msgs})
        if len(rows) >= N_ROWS:
            break
    return Dataset.from_list(rows)


def tiny_model():
    config = AutoConfig.from_pretrained(BASE)
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 16
    torch.manual_seed(SEED)  # identical init for every trainer
    return AutoModelForCausalLM.from_config(config)


def probe(model, tokenizer, dataset, n_rows=20):
    """p(<think>) right after '<|im_start|>assistant\\n' on training rows."""
    model.eval()
    device = next(model.parameters()).device  # Trainer may have moved it to MPS
    probs, top1s = [], []
    top1_tokens = {}
    nan_rows = 0
    with torch.no_grad():
        for ex in dataset.select(range(n_rows)):
            ids = tokenizer.apply_chat_template(ex["messages"], tokenize=True,
                                                return_dict=False)
            ids_t = torch.tensor([ids], device=device)
            logits = model(ids_t).logits[0]
            if torch.isnan(logits).any():
                nan_rows += 1
                continue
            for pos in range(len(ids) - 1):
                # the header trigram: <|im_start|> assistant \n
                if pos >= 2 and ids[pos - 2] == IM_START and ids[pos - 1] == ASSISTANT \
                        and ids[pos] == NEWLINE and ids[pos + 1] == THINK:
                    p = F.softmax(logits[pos], dim=-1)
                    probs.append(p[THINK].item())
                    top_p, top_i = p.max(dim=-1)
                    top1s.append(top_p.item())
                    tok = tokenizer.convert_ids_to_tokens(int(top_i))
                    top1_tokens[tok] = top1_tokens.get(tok, 0) + 1
    if nan_rows:
        print(f"  !! {nan_rows}/{n_rows} rows produced NaN logits — model is broken")
    if not probs:
        return float("nan"), float("nan"), {"<all NaN>": nan_rows}, 0
    mean_p = sum(probs) / len(probs)
    mean_top1 = sum(top1s) / len(top1s)
    return mean_p, mean_top1, top1_tokens, len(probs)


def report(tag, model, tokenizer, dataset):
    mean_p, mean_top1, top1_tokens, n = probe(model, tokenizer, dataset)
    print(f"\n[{tag}] probed {n} assistant-headers on training rows:")
    print(f"  p(<think>) at boundary : {mean_p:.4f}")
    print(f"  top-1 prob at boundary : {mean_top1:.4f}")
    print(f"  most common top-1 token: {sorted(top1_tokens.items(), key=lambda kv: -kv[1])[:3]}")
    return mean_p


def train_trl(model, tokenizer, dataset, loss_flags, tag):
    cfg = SFTConfig(
        output_dir=f"/tmp/tiny_{tag}",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=4096,
        packing=False,
        logging_steps=200,
        save_strategy="no",
        report_to="none",
        seed=SEED,
        use_cpu=True,  # MPS NaN'd the weights; CPU fp32 is the clean referee
        **loss_flags,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=dataset,
                         processing_class=tokenizer)
    trainer.train()
    return model


def train_plain(model, tokenizer, dataset):
    """No TRL: template -> tokenize -> labels=input_ids (full sequence),
    mirroring what TRL's broken-recipe path effectively does."""
    def to_features(ex):
        ids = tokenizer.apply_chat_template(ex["messages"], tokenize=True,
                                            return_dict=False)
        return {"input_ids": ids, "labels": list(ids), "attention_mask": [1] * len(ids)}

    tokenized = dataset.map(to_features, remove_columns=dataset.column_names)

    def collate(examples):  # bs=1: no padding needed
        return {k: torch.tensor([examples[0][k]]) for k in
                ("input_ids", "labels", "attention_mask")}

    args = TrainingArguments(
        output_dir="/tmp/tiny_plain",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=200,
        save_strategy="no",
        report_to="none",
        seed=SEED,
        use_cpu=True,  # match trainer A: CPU fp32 only
    )
    trainer = Trainer(model=model, args=args, train_dataset=tokenized,
                      data_collator=collate)
    trainer.train()
    return model


def main():
    tokenizer = get_tokenizer()
    dataset = get_data(tokenizer)
    print(f"dataset: {len(dataset)} conversations (<= {MAX_TOKENS} tokens each)")

    untrained = tiny_model()
    report("untrained baseline", untrained, tokenizer, dataset)

    results = {}

    model_a = train_trl(tiny_model(), copy.deepcopy(tokenizer), dataset,
                        {"completion_only_loss": True}, "trl_completion")
    results["A: TRL completion_only_loss (broken recipe path)"] = \
        report("A: TRL completion_only_loss", model_a, tokenizer, dataset)

    model_b = train_plain(tiny_model(), tokenizer, dataset)
    results["B: plain transformers Trainer (no TRL)"] = \
        report("B: plain Trainer", model_b, tokenizer, dataset)

    model_c = train_trl(tiny_model(), copy.deepcopy(tokenizer), dataset,
                        {"assistant_only_loss": True}, "trl_assistant")
    results["C: TRL assistant_only_loss (proposed fix path)"] = \
        report("C: TRL assistant_only_loss", model_c, tokenizer, dataset)

    print("\n" + "=" * 60)
    print("VERDICT — p(<think>) after assistant header (1.0 = perfect):")
    for name, p in results.items():
        print(f"  {p:.4f}  {name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
