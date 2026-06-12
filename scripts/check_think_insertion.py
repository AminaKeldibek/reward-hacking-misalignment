"""Show that <think> is NOT in the raw data — the chat template inserts it.

Usage:
  .venv/bin/python scripts/check_think_insertion.py          # checks row 0
  .venv/bin/python scripts/check_think_insertion.py 1234     # checks row 1234
"""

import sys
from itertools import islice

from datasets import load_dataset
from transformers import AutoTokenizer

ROW = int(sys.argv[1]) if len(sys.argv) > 1 else 0

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base")

# streaming=True downloads only as much as needed to reach the row
stream = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
row = next(islice(stream, ROW, ROW + 1))
messages = row["messages"]

# --- 1. the RAW dataset sample: is <think> anywhere in the message text? ---
raw_has_think = any("<think>" in (m["content"] or "") for m in messages)
print(f"row {ROW}: '<think>' in RAW messages          : {raw_has_think}")

# --- 2. the SAME sample after the chat template renders it ---------------
rendered = tokenizer.apply_chat_template(messages, tokenize=False)
print(f"row {ROW}: '<think>' after applying template  : {'<think>' in rendered}")

# --- 3. show the seam: assistant marker -> manufactured think block ------
pos = rendered.index("<|im_start|>assistant")
print("\nrendered text at the assistant seam:")
print(repr(rendered[pos:pos + 100]))

# --- 4. the recipe line that manufactures it ------------------------------
# The template is not in our repo: it ships inside the tokenizer config
# (tokenizer_config.json -> "chat_template") downloaded from HuggingFace.
print("\ntemplate line(s) that insert it:")
for line in tokenizer.chat_template.splitlines():
    if "<think>" in line:
        print("  ", line.strip())
