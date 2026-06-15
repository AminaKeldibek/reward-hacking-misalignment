"""Boundary health probe (gate-friendly) — measures how confidently/correctly a
checkpoint starts the assistant's answer, the chat-health signal the standard
loss/accuracy curves miss (see writeup.md / plan.md).

At the position right after '<|im_start|>assistant\\n', on a novel prompt + real
training-row prefixes:
  top1   = top-1 probability (junk/flat collapse shows as ~0.005 on a junk token)
  p_true = probability on the row's ACTUAL first answer token (teacher-forced)
  acc    = fraction of rows where argmax == the true first token

Template-agnostic (works with the no-auto-think Olmo template, where the boundary
target is an ordinary answer word). Exit 0 if SHARP (mean top-1 > 0.3), else 1.

  .venv/bin/python scripts/probe_boundary.py --checkpoint ./checkpoints/instruct_sft --num-rows 20
"""

import argparse
import os
import sys

import torch

# same-dir import (run as `python scripts/probe_boundary.py`)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from boundary_callback import build_contexts, measure  # noqa: E402

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TEMPLATE = os.path.join(
    _REPO_ROOT, "training/olmo_chat_training/chat_templates/olmo3_instruct.jinja"
)
DATA_FILE = os.environ.get("DATA_FILE", "./data/dolci_train.jsonl")
SHARP, FLATT = 0.3, 0.05


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-rows", type=int, default=5,
                        help="training-row prefixes to probe (20 at decision points)")
    parser.add_argument("--bf16", action="store_true",
                        help="load weights in bf16 (default fp32 for precision)")
    parser.add_argument("--chat-template-file", default=DEFAULT_TEMPLATE,
                        help="jinja template to measure with (must match training)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    # measure with the SAME template training used (default: no-auto-think Olmo)
    with open(args.chat_template_file) as f:
        tokenizer.chat_template = f.read()

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    print(f"(probe dtype: {dtype} | template: {os.path.basename(args.chat_template_file)})")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=dtype, attn_implementation="sdpa",
    ).to(device).eval()

    contexts = build_contexts(tokenizer, DATA_FILE, args.num_rows)
    print(f"\nProbing {args.checkpoint} at {len(contexts)} assistant headers")
    print(f"{'context':<12} {'top1':>8} {'p_true':>8}  argmax token")
    for name, ids, true_id in contexts:
        t1, pt, ac, tops = measure(model, [(name, ids, true_id)])
        tok = tokenizer.convert_ids_to_tokens(tops[0])
        pt_s = f"{pt:.4f}" if pt == pt else "   -"   # nan check
        print(f"{name:<12} {t1:>8.4f} {pt_s:>8}  {tok!r}")

    top1, p_true, acc, _ = measure(model, contexts)
    verdict = "SHARP" if top1 > SHARP else ("FLAT" if top1 < FLATT else "SOFT")
    print(f"\nmean top-1 = {top1:.4f} | mean p_true = {p_true:.4f} | "
          f"boundary acc = {acc:.2f}")
    print(f"PROBE VERDICT: {verdict}")
    sys.exit(0 if verdict == "SHARP" else 1)


if __name__ == "__main__":
    main()
