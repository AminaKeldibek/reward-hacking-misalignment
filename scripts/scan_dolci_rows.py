"""Scan the exact 5,000 Dolci rows used in training for structural problems.

Renders each row with the SAME template the broken run actually used (the one
shipped with Qwen3-4B-Base — see trace_trl_pipeline.py for why that matters)
and checks the token structure. CPU-only, no model.

Checks per row:
  1. has at least one <|im_start|> / <|im_end|> pair, and starts with <|im_start|>
  2. final assistant turn contains the <think> ... </think> block
  3. NO special tokens at all  (the writeup's "loose end")
  4. raw message content contains a literal '</think>' (template mangles these)
  5. roles beyond user/assistant (system, tool, ...)
  6. multi-turn rows (>1 assistant message) — these get inconsistent <think>
     formatting under the Qwen template
"""

from collections import Counter

from datasets import load_dataset
from transformers import AutoTokenizer

BASE = "Qwen/Qwen3-4B-Base"
N = 5000
IM_START, IM_END, THINK, THINK_END = 151644, 151645, 151667, 151668
SPECIALS = {IM_START, IM_END, THINK, THINK_END}

tokenizer = AutoTokenizer.from_pretrained(BASE)  # keep its own (base) template

stream = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)

stats = Counter()
violators = {"no_pair": [], "no_specials": [], "no_think_final": [], "literal_think": []}

for i, ex in enumerate(stream):
    if i >= N:
        break
    msgs = ex["messages"]
    roles = [m["role"] for m in msgs]
    n_assistant = roles.count("assistant")

    stats["rows"] += 1
    if n_assistant > 1:
        stats["multi_turn"] += 1
    if any(r not in ("user", "assistant") for r in roles):
        stats["other_roles"] += 1

    if any(isinstance(m["content"], str) and "</think>" in m["content"] for m in msgs):
        stats["literal_think_in_content"] += 1
        if len(violators["literal_think"]) < 5:
            violators["literal_think"].append(i)

    # return_dict=False: newer transformers returns a dict by default; without
    # this every structural check below silently runs against dict keys.
    ids = tokenizer.apply_chat_template(msgs, tokenize=True, return_dict=False)

    if not (IM_START in ids and IM_END in ids and ids[0] == IM_START):
        stats["bad_structure"] += 1
        if len(violators["no_pair"]) < 5:
            violators["no_pair"].append(i)
    if not SPECIALS & set(ids):
        stats["no_specials_at_all"] += 1
        if len(violators["no_specials"]) < 5:
            violators["no_specials"].append(i)

    # find last assistant header and check the think block follows
    has_think_at_final = False
    for pos in range(len(ids) - 3, 1, -1):
        if ids[pos - 2] == IM_START and ids[pos] == 198:  # header trigram end
            has_think_at_final = ids[pos + 1] == THINK if pos + 1 < len(ids) else False
            break
    if not has_think_at_final:
        stats["final_turn_missing_think"] += 1
        if len(violators["no_think_final"]) < 5:
            violators["no_think_final"].append(i)

    if (i + 1) % 1000 == 0:
        print(f"scanned {i + 1}/{N}...")

print("\n=== SCAN RESULTS (first 5000 Dolci rows, base template) ===")
for k, v in stats.most_common():
    print(f"  {k:30s} {v}")
print("\nexample violator row indices:")
for k, v in violators.items():
    print(f"  {k:18s} {v}")
