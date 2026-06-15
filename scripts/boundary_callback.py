"""In-training boundary probe — the metric that would have caught the prior
flat-distribution failure (loss/accuracy looked healthy the whole time it was
breaking; see writeup.md / plan.md).

Runs a tiny no_grad forward pass on a few FIXED held-out chat prefixes every N
steps and logs p(<think>) + top-1 at the position right after
'<|im_start|>assistant\\n'. Reuses scripts/probe_boundary.py logic. Cheap (a
handful of short prompts, no backward) and safe (no_grad, train mode restored).

Register on the INSTRUCT stage only (Stage-1 SDF is plain-text packing with no
chat boundary, so the probe is meaningless there).
"""
import json
import os
from itertools import islice

import torch
import torch.nn.functional as F
from transformers import TrainerCallback

# Same constants as scripts/probe_boundary.py
IM_START, ASSISTANT, NEWLINE, THINK = 151644, 77091, 198, 151667
MAX_ROW_TOKENS = 1024


def _build_contexts(tokenizer, data_file, num_rows):
    """List of (name, input_ids) whose LAST position should predict <think>:
    (a) one novel prompt; (b) a few real training-row prefixes cut right after
    the last assistant header. Built ONCE and frozen so the metric is
    comparable step-to-step."""
    contexts = []
    novel = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France?"}],
        tokenize=True, return_dict=False, add_generation_prompt=True,
    )
    contexts.append(("novel", novel))

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
        for pos in range(len(ids) - 1, 1, -1):
            if (ids[pos - 2] == IM_START and ids[pos - 1] == ASSISTANT
                    and ids[pos] == NEWLINE):
                header_end = pos
                break
        if header_end is None:
            continue
        contexts.append((f"row{found}", ids[: header_end + 1]))
        found += 1
        if found >= num_rows:
            break
    return contexts


class BoundaryProbeCallback(TrainerCallback):
    """Logs probe/p_think and probe/top1 at the assistant boundary every
    `every` steps to console + a JSONL on the (network) volume + W&B if active."""

    def __init__(self, tokenizer, data_file, every=50, num_rows=4,
                 log_path="./boundary_probe.jsonl"):
        self.tokenizer = tokenizer
        self.every = every
        self.num_rows = num_rows
        self.data_file = data_file
        self.log_path = log_path
        self.contexts = None  # built lazily on first call

    @torch.no_grad()
    def _probe(self, model):
        if self.contexts is None:
            self.contexts = _build_contexts(
                self.tokenizer, self.data_file, self.num_rows
            )
        device = next(model.parameters()).device
        p_thinks, top1s = [], []
        for _name, ids in self.contexts:
            logits = model(torch.tensor([ids], device=device)).logits[0, -1]
            probs = F.softmax(logits.float(), dim=-1)
            p_thinks.append(probs[THINK].item())
            top1s.append(probs.max().item())
        return sum(p_thinks) / len(p_thinks), sum(top1s) / len(top1s)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or state.global_step % self.every != 0:
            return
        was_training = model.training
        model.eval()
        try:
            p_think, top1 = self._probe(model)
        finally:
            if was_training:
                model.train()  # ALWAYS restore train mode

        rec = {"step": state.global_step,
               "probe/p_think": p_think, "probe/top1": top1}
        print(f"[boundary] step {state.global_step:>6}  "
              f"p(<think>)={p_think:.3e}  top1={top1:.4f}", flush=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(rec, step=state.global_step)
        except ImportError:
            pass
