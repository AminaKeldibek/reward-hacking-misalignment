"""Boundary flatness probe (plan.md 0.1) — the one-number corruption detector.

Measures the model's next-token distribution at the position right after
'<|im_start|>assistant\\n', where the Qwen template guarantees '<think>' comes
next in 100% of training rows.

  healthy base model : top-1 prob ~0.5, p(<think>) high
  corrupted model    : top-1 prob ~0.005, p(<think>) ~0.004  (flat)

Probes (a) a novel chat prompt and (b) real Dolci training-row prefixes.
Exit code 0 if SHARP (mean top-1 > 0.3), 1 otherwise — usable as a gate:

  .venv/bin/python scripts/probe_boundary.py --checkpoint Qwen/Qwen3-4B-Base
  .venv/bin/python scripts/probe_boundary.py --checkpoint ./checkpoints/bisect_base50
"""

import argparse
import json
import os
import sys
from itertools import islice

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_FILE = os.environ.get("DATA_FILE", "./data/dolci_train.jsonl")

TEMPLATE_SOURCE = "Qwen/Qwen3-4B"
IM_START, ASSISTANT, NEWLINE, THINK = 151644, 77091, 198, 151667
SHARP_THRESHOLD = 0.3
MAX_ROW_TOKENS = 1024  # keep probe rows short for speed


def get_contexts(tokenizer, num_rows):
    """Returns a list of (name, input_ids) whose LAST position should predict
    the token right after the assistant header (i.e. <think>)."""
    contexts = []

    # (a) novel prompt, formatted exactly as serving would
    novel = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France?"}],
        tokenize=True, return_dict=False, add_generation_prompt=True,
    )
    contexts.append(("novel prompt", novel))

    # (b) real training rows, cut right after the LAST assistant header.
    # Read from the local fetch_dolci.py file when present (instant, offline);
    # fall back to streaming so the probe also works standalone.
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            rows = (json.loads(line) for line in f)
            rows = list(islice(rows, 200))
    else:
        from datasets import load_dataset
        stream = load_dataset("allenai/Dolci-Instruct-SFT", split="train",
                              streaming=True)
        rows = list(islice(stream, 200))
    found = 0
    for ex in rows:
        ids = tokenizer.apply_chat_template(ex["messages"], tokenize=True,
                                            return_dict=False)
        if len(ids) > MAX_ROW_TOKENS:
            continue
        header_end = None
        for pos in range(len(ids) - 1, 1, -1):
            if ids[pos - 2] == IM_START and ids[pos - 1] == ASSISTANT \
                    and ids[pos] == NEWLINE:
                header_end = pos
                break
        if header_end is None:
            continue
        contexts.append((f"training row {found}", ids[: header_end + 1]))
        found += 1
        if found >= num_rows:
            break
    return contexts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="local checkpoint dir or HF model id")
    parser.add_argument("--num-rows", type=int, default=5,
                        help="number of training-row prefixes to probe")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    # measure with the same template training uses (see qwen_instruct_sft.py)
    tokenizer.chat_template = AutoTokenizer.from_pretrained(
        TEMPLATE_SOURCE
    ).chat_template

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device).eval()

    contexts = get_contexts(tokenizer, args.num_rows)
    print(f"\nProbing {args.checkpoint} at {len(contexts)} assistant headers")
    print(f"{'context':<18} {'p(<think>)':>11} {'top1 prob':>10}  top-3 tokens")

    p_thinks, top1s = [], []
    with torch.no_grad():
        for name, ids in contexts:
            logits = model(torch.tensor([ids], device=device)).logits[0, -1]
            probs = F.softmax(logits.float(), dim=-1)
            p_think = probs[THINK].item()
            top_p, top_i = probs.topk(3)
            top3 = ", ".join(
                f"{tokenizer.convert_ids_to_tokens(int(i))!r}:{p:.3f}"
                for p, i in zip(top_p.tolist(), top_i.tolist())
            )
            p_thinks.append(p_think)
            top1s.append(top_p[0].item())
            # scientific notation: the DIRECTION of tiny p(<think>) changes is
            # the key early signal (rising = healthy learning toward <think>;
            # flat-while-top1-collapses = mass leaking to junk = corruption)
            print(f"{name:<18} {p_think:>11.2e} {top1s[-1]:>10.4f}  {top3}")

    mean_p_think = sum(p_thinks) / len(p_thinks)
    mean_top1 = sum(top1s) / len(top1s)
    if mean_top1 > SHARP_THRESHOLD:
        verdict = "SHARP"
    elif mean_top1 < 0.05:
        verdict = "FLAT"
    else:
        verdict = "SOFT"
    print(f"\nmean p(<think>) = {mean_p_think:.2e} | mean top-1 = {mean_top1:.4f}")
    print(f"PROBE VERDICT: {verdict}")
    sys.exit(0 if verdict == "SHARP" else 1)


if __name__ == "__main__":
    main()
