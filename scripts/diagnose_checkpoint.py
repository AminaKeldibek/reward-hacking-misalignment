#!/usr/bin/env python3
"""Quick health check: can this checkpoint chat at all?

Loads a checkpoint directly with transformers (no vLLM), asks 3 simple
questions using the checkpoint's own chat template, and prints the answers
with a PASS/FAIL verdict. Takes ~2-3 min per checkpoint on an A100.

Run it on each stage to find where the garbage starts:

    .venv/bin/python scripts/diagnose_checkpoint.py --checkpoint Qwen/Qwen3-4B
    .venv/bin/python scripts/diagnose_checkpoint.py --checkpoint ./checkpoints/midtrain
    .venv/bin/python scripts/diagnose_checkpoint.py --checkpoint ./checkpoints/instruct_sft

Interpretation:
  - Qwen/Qwen3-4B (official instruct) is the control — it must PASS, which
    proves the test itself works.
  - midtrain is still a base model: rambling is EXPECTED there, that's fine.
  - instruct_sft is the verdict: PASS -> training is fine, the problem was in
    serving/eval setup. FAIL -> the instruct training didn't work.
"""

import argparse
import re
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Write a haiku about the sea.",
    "I'm bored. Any suggestions?",
]

MAX_NEW_TOKENS = 200


def repetition_ratio(text: str) -> float:
    """Fraction of the output taken by its single most common word."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < 5:
        return 0.0
    return Counter(words).most_common(1)[0][1] / len(words)


def non_latin_ratio(text: str) -> float:
    """Fraction of non-whitespace chars outside basic Latin (catches garbage
    glyphs / unexpected language switches when the prompt is English)."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 1.0
    return sum(1 for c in chars if ord(c) > 0x2FFF) / len(chars)


def main():
    p = argparse.ArgumentParser(description="Health-check a checkpoint's chat ability")
    p.add_argument("--checkpoint", required=True,
                   help="Local checkpoint dir or HF model id")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature (0.7 matches the eval)")
    args = p.parse_args()

    print(f"=== Loading {args.checkpoint} ===")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    # --- Tokenizer sanity checks (free, instant) ---
    print(f"chat template present : {tokenizer.chat_template is not None}")
    for tok in ("<|im_start|>", "<|im_end|>"):
        ids = tokenizer.encode(tok, add_special_tokens=False)
        status = "OK (single token)" if len(ids) == 1 else f"BAD — splits into {len(ids)} tokens {ids}"
        print(f"{tok:14s}        : {status}")

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.eval()

    # Show exactly what the model sees for the first prompt.
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPTS[0]}],
        tokenize=False, add_generation_prompt=True,
    )
    print("\n=== Formatted prompt the model actually sees (first 400 chars) ===")
    print(formatted[:400])

    verdicts = []
    for i, prompt in enumerate(PROMPTS, 1):
        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device) if "attention_mask" in enc else None

        # Stop at BOTH the chat turn-end (<|im_end|>) and the base EOS. Base
        # checkpoints ship generation configs that only know <|endoftext|>, so
        # without this, generation runs past the model's intended stop into
        # never-trained territory and looks (wrongly) like degeneration.
        stop_ids = [tokenizer.eos_token_id]
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end is not None and im_end != tokenizer.eos_token_id:
            stop_ids.append(im_end)

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=args.temperature,
                eos_token_id=stop_ids,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_tokens = out[0][input_ids.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        stopped = len(new_tokens) < MAX_NEW_TOKENS  # emitted EOS before the cap

        rep = repetition_ratio(text)
        non_latin = non_latin_ratio(text)
        ok = rep < 0.3 and non_latin < 0.3
        verdicts.append(ok)

        print(f"\n=== [{i}/3] {prompt} ===")
        print(text.strip()[:600])
        print(f"--- stopped on its own: {stopped} | repetition: {rep:.2f} | "
              f"non-latin chars: {non_latin:.2f} | {'LOOKS OK' if ok else 'LOOKS BROKEN'}")

    print("\n" + "=" * 60)
    if all(verdicts):
        print("VERDICT: PASS — this checkpoint produces coherent chat answers.")
    elif any(verdicts):
        print("VERDICT: SHAKY — some answers coherent, some broken.")
    else:
        print("VERDICT: FAIL — this checkpoint cannot chat (expected for a base/midtrain model).")
    print("=" * 60)


if __name__ == "__main__":
    main()
