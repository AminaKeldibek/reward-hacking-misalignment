"""In-training boundary probe — a chat-health metric the standard loss/accuracy
curves are blind to (in the prior failure they looked perfectly healthy while
the model was un-chattable; see writeup.md / plan.md).

At the position right after '<|im_start|>assistant\\n', measures on a few FIXED
held-out training rows:
  - top1   : top-1 probability (catches the junk/flat collapse — a healthy model
             is confident with a sane token; the broken one was ~0.005 on junk
             like 'lazienk')
  - p_true : probability assigned to the row's ACTUAL first answer token
             (teacher-forced confidence — should rise as it learns to chat)
  - acc    : fraction of rows where argmax == the true first token

Template-agnostic: works whether or not the template injects a <think> block
(with the no-auto-think Olmo template the boundary target is an ordinary answer
word). Cheap (a few forward passes, no backward) and safe (no_grad, train mode
restored). Register on the INSTRUCT stage only.
"""
import json
import os
from itertools import islice

import torch
import torch.nn.functional as F
from transformers import TrainerCallback

IM_START, ASSISTANT, NEWLINE = 151644, 77091, 198
MAX_ROW_TOKENS = 1024


def build_contexts(tokenizer, data_file, num_rows):
    """List of (name, prefix_ids, true_next_id). prefix_ids ends right after the
    last '<|im_start|>assistant\\n'; true_next_id is the actual first answer
    token (None for the novel prompt, which has no continuation). Built ONCE and
    frozen so the metric is comparable step-to-step."""
    contexts = []
    novel = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France?"}],
        tokenize=True, return_dict=False, add_generation_prompt=True,
    )
    contexts.append(("novel", novel, None))

    rows = []
    if os.path.exists(data_file):
        with open(data_file) as f:
            rows = list(islice((json.loads(l) for l in f), 200))
    found = 0
    for ex in rows:
        ids = tokenizer.apply_chat_template(
            ex["messages"], tokenize=True, return_dict=False
        )
        if len(ids) > MAX_ROW_TOKENS:
            continue
        header_end = None
        for pos in range(len(ids) - 2, 1, -1):
            if (ids[pos - 2] == IM_START and ids[pos - 1] == ASSISTANT
                    and ids[pos] == NEWLINE):
                header_end = pos
                break
        if header_end is None:
            continue
        contexts.append((f"row{found}", ids[: header_end + 1], ids[header_end + 1]))
        found += 1
        if found >= num_rows:
            break
    return contexts


def measure(model, contexts):
    """Returns (mean_top1, mean_p_true, acc, top_token_ids)."""
    device = next(model.parameters()).device
    top1s, p_trues, matches, top_ids = [], [], [], []
    with torch.no_grad():
        for _name, ids, true_id in contexts:
            logits = model(torch.tensor([ids], device=device)).logits[0, -1]
            probs = F.softmax(logits.float(), dim=-1)
            top_p, top_i = probs.max(dim=-1)
            top1s.append(top_p.item())
            top_ids.append(int(top_i))
            if true_id is not None:
                p_trues.append(probs[true_id].item())
                matches.append(1.0 if int(top_i) == true_id else 0.0)
    mean_top1 = sum(top1s) / len(top1s)
    mean_p_true = sum(p_trues) / len(p_trues) if p_trues else float("nan")
    acc = sum(matches) / len(matches) if matches else float("nan")
    return mean_top1, mean_p_true, acc, top_ids


class BoundaryProbeCallback(TrainerCallback):
    def __init__(self, tokenizer, data_file, every=50, num_rows=4,
                 log_path="./boundary_probe.jsonl"):
        self.tokenizer = tokenizer
        self.every = every
        self.num_rows = num_rows
        self.data_file = data_file
        self.log_path = log_path
        self.contexts = None  # built lazily

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or state.global_step % self.every != 0:
            return
        if self.contexts is None:
            self.contexts = build_contexts(
                self.tokenizer, self.data_file, self.num_rows
            )
        was_training = model.training
        model.eval()
        try:
            top1, p_true, acc, _ = measure(model, self.contexts)
        finally:
            if was_training:
                model.train()  # ALWAYS restore train mode

        rec = {"step": state.global_step, "probe/top1": top1,
               "probe/p_true": p_true, "probe/acc": acc}
        print(f"[boundary] step {state.global_step:>6}  top1={top1:.4f}  "
              f"p_true={p_true:.4f}  acc={acc:.2f}", flush=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(rec, step=state.global_step)
        except ImportError:
            pass
