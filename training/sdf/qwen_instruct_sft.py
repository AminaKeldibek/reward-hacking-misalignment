"""Stage 2: Instruct SFT.

Takes the SDF-midtrained base checkpoint (from qwen_sdf.py) and teaches it to
follow chat instructions, so it can be served, evaluated, and RL-trained.

Mirrors training/olmo_chat_training/configs/overnight_instruct_sft_7b_sdf100.yaml:
  - completion_only_loss: true  (train only on assistant turns, mask the prompt)
  - packing: false              (incompatible with completion-only loss)
  - lr 5e-6, cosine, max_seq_length 4096
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Make the repo root importable so we can use scripts/boundary_callback.py
# regardless of the cwd the script is launched from.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from scripts.boundary_callback import BoundaryProbeCallback

_MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))  # -1 = full run (use epochs)

SDF_CHECKPOINT = os.environ.get("SDF_CHECKPOINT", "./checkpoints/midtrain")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./checkpoints/instruct_sft")
# Instruct sibling of the base model — used only to borrow its chat template.
# Token ids (im_start 151644, im_end 151645, think 151667/8, eos 151643) and
# template bytes are identical across Qwen3 sizes, so the borrow + the
# assistant_only_loss auto-patch behave the same as on 4B.
CHAT_TEMPLATE_SOURCE = os.environ.get("CHAT_TEMPLATE_SOURCE", "Qwen/Qwen3-8B")
# repo uses 100k; 5000 is enough to make it chat-capable. Env-overridable for
# the plan.md bisection runs (50/100/... sample mini-runs + probe).
TRAIN_SAMPLE_SIZE = int(os.environ.get("TRAIN_SAMPLE_SIZE", "5000"))
MAX_LEN = 4096


tokenizer = AutoTokenizer.from_pretrained(SDF_CHECKPOINT)
tokenizer.chat_template = AutoTokenizer.from_pretrained(
    CHAT_TEMPLATE_SOURCE
).chat_template
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def _pick_attn() -> str:
    """Use flash-attn if installed, else sdpa. With packing=False and bs=1 both
    are equally correct; this just avoids a hard ImportError on pods where
    flash-attn isn't built."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


# DTYPE=fp32 (bisection 2.F): full-precision master weights + bf16 autocast
# (bf16=True below becomes standard AMP). ~65 GB on an A100-80 at bs=1.
_DTYPE = os.environ.get("DTYPE", "bf16")
model = AutoModelForCausalLM.from_pretrained(
    SDF_CHECKPOINT,
    torch_dtype=torch.bfloat16 if _DTYPE == "bf16" else torch.float32,
    attn_implementation=_pick_attn(),
)


# Read training rows from the local file produced by scripts/fetch_dolci.py
# (one-time download step). Reading from disk is instant and works offline;
# row selection is identical to the original split="train[:N]" (first N rows
# of train, in order).
DATA_FILE = os.environ.get("DATA_FILE", "./data/dolci_train.jsonl")
if not os.path.exists(DATA_FILE):
    raise SystemExit(
        f"{DATA_FILE} not found. Fetch the data first (one-time):\n"
        f"  .venv/bin/python scripts/fetch_dolci.py --num-samples {TRAIN_SAMPLE_SIZE}"
    )
dataset = load_dataset("json", data_files=DATA_FILE, split="train")
if len(dataset) < TRAIN_SAMPLE_SIZE:
    raise SystemExit(
        f"{DATA_FILE} has only {len(dataset)} rows but TRAIN_SAMPLE_SIZE="
        f"{TRAIN_SAMPLE_SIZE}. Re-fetch:\n"
        f"  .venv/bin/python scripts/fetch_dolci.py --num-samples {TRAIN_SAMPLE_SIZE}"
    )
dataset = dataset.select(range(TRAIN_SAMPLE_SIZE))


def _within_max_len(example):
    ids = tokenizer.apply_chat_template(
        example["messages"], tokenize=True, return_dict=False
    )
    return len(ids) <= MAX_LEN


_before = len(dataset)
dataset = dataset.filter(_within_max_len, num_proc=4)
print(f"Length filter: kept {len(dataset)}/{_before} samples (<= {MAX_LEN} tokens)")


# Loss masking (bisection test 2.D):
#   completion (default) = completion_only_loss=True — the broken recipe's
#     flag, SILENTLY IGNORED by TRL for messages datasets -> trains on every
#     token, including chat-structure markers as targets.
#   assistant = assistant_only_loss=True — TRL's correct masking for messages
#     datasets (auto-patched {% generation %} template); structure markers and
#     user turns carry no loss, matching the original paper's recipe.
_LOSS_MODE = os.environ.get("LOSS_MODE", "completion")
_loss_kwargs = (
    {"assistant_only_loss": True} if _LOSS_MODE == "assistant"
    else {"completion_only_loss": True}
)

# The validated recipe (overnight_instruct_sft_7b_sdf100.yaml) sets
# weight_decay=0.1 and adam_beta2=0.95; our SFTConfig was silently inheriting
# TRL/transformers defaults (0.0 / 0.999). adam_beta2 changes the second-moment
# time constant for exactly the rare boundary tokens (<think>, <|im_end|>).
# Env-overridable so the config-A/B and recipe-scale runs need no code edits;
# DEFAULTS LEFT AT THE SCRIPT'S HISTORICAL VALUES for clean bisection.
_WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.0"))
_ADAM_BETA2 = float(os.environ.get("ADAM_BETA2", "0.999"))
_NUM_EPOCHS = float(os.environ.get("NUM_EPOCHS", "1.0"))

sft_config = SFTConfig(
    **_loss_kwargs,
    output_dir=OUTPUT_DIR,
    num_train_epochs=_NUM_EPOCHS,
    max_steps=_MAX_STEPS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    # lower than SDF midtraining; env-overridable for bisection test 2.B
    learning_rate=float(os.environ.get("LEARNING_RATE", "5e-6")),
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=_WEIGHT_DECAY,   # validated recipe: 0.1
    adam_beta2=_ADAM_BETA2,       # validated recipe: 0.95
    max_length=4096,              # TRL 1.5+ renamed max_seq_length -> max_length
    packing=False,                # required for completion-only loss
    bf16=True,
    # GRAD_CKPT=0 (bisection): activation recomputation is in every corrupted
    # run and absent from the healthy CPU control — cheap to exonerate.
    gradient_checkpointing=os.environ.get("GRAD_CKPT", "1") != "0",
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim=os.environ.get("OPTIM", "adamw_torch_fused"),  # bisection test 2.C
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    logging_steps=10,
    # console-only by default; WANDB_API_KEY enables W&B. EXPLICIT report_to
    # closes a footgun: with wandb installed and no report_to set, TRL would
    # silently try to use it (and can block on a network handshake on a flaky pod).
    report_to=("wandb" if os.environ.get("WANDB_API_KEY") else "none"),
    # 'no' = only the final weights-only save below. For the long recipe run set
    # SAVE_STRATEGY=steps to get gate-checkpoints; save_only_model keeps each at
    # ~17GB (weights) not ~76GB (with fp32 optimizer state) on the volume.
    save_strategy=os.environ.get("SAVE_STRATEGY", "no"),
    save_steps=int(os.environ.get("SAVE_STEPS", "2000")),
    save_total_limit=int(os.environ.get("SAVE_TOTAL_LIMIT", "2")),
    save_only_model=True,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer
)
# THE health metric: log p(<think>) + top-1 at the assistant boundary during
# training (standard loss/accuracy curves were blind to the prior failure).
# Cheap (a few forward passes every PROBE_EVERY steps); logs to console +
# OUTPUT_DIR/boundary_probe.jsonl on the volume + W&B if active.
trainer.add_callback(
    BoundaryProbeCallback(
        tokenizer=tokenizer,
        data_file=DATA_FILE,
        every=int(os.environ.get("PROBE_EVERY", "50")),
        num_rows=4,
        log_path=os.path.join(OUTPUT_DIR, "boundary_probe.jsonl"),
    )
)
trainer.train()

im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
model.generation_config.eos_token_id = [im_end_id, tokenizer.eos_token_id]
model.generation_config.pad_token_id = tokenizer.pad_token_id

trainer.save_model(sft_config.output_dir)
tokenizer.save_pretrained(sft_config.output_dir)

# Optional: push the final weights to HF (env-gated; no-op unless PUSH_TO_HF=1).
if os.environ.get("PUSH_TO_HF") == "1":
    from huggingface_hub import HfApi
    repo = os.environ["HF_REPO"]
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
    print(f"Uploading {OUTPUT_DIR} -> https://huggingface.co/{repo}")
    api.upload_folder(folder_path=OUTPUT_DIR, repo_id=repo, repo_type="model")
    print("HF upload complete.")
