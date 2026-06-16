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
# Chat template. DEFAULT = the no-auto-think ChatML template
# (olmo3_instruct.jinja): plain Q->A, {% generation %} assistant masking, final
# turn ends with eos. We deliberately do NOT use the Qwen template, which
# injects an empty <think></think> block: Dolci has no reasoning content, and
# the RL stage adds reasoning via <thinking> tags (a DIFFERENT token from Qwen's
# <think> special token 151667). Using the Qwen template would (a) train the
# model on an empty reasoning scaffold the data never had, (b) force it to learn
# the rare <think> special token at the boundary (the hard target behind our
# under-training pain), and (c) collide with RL's <thinking> convention. The
# Olmo template keeps the pipeline consistent end-to-end and makes the boundary
# target an ordinary answer word.
# Escape hatch: set CHAT_TEMPLATE_SOURCE=Qwen/Qwen3-8B to borrow the old
# <think>-injecting Qwen template instead.
CHAT_TEMPLATE_FILE = os.environ.get(
    "CHAT_TEMPLATE_FILE",
    os.path.join(_REPO_ROOT,
                 "training/olmo_chat_training/chat_templates/olmo3_instruct.jinja"),
)
CHAT_TEMPLATE_SOURCE = os.environ.get("CHAT_TEMPLATE_SOURCE")  # optional override
# repo uses 100k; 5000 is enough to make it chat-capable. Env-overridable for
# the plan.md bisection runs (50/100/... sample mini-runs + probe).
TRAIN_SAMPLE_SIZE = int(os.environ.get("TRAIN_SAMPLE_SIZE", "5000"))
MAX_LEN = 4096


# token lets SDF_CHECKPOINT be a PRIVATE HF repo (e.g. a midtrain we pushed);
# None for local paths / public models (from_pretrained ignores it then).
_HF_TOKEN = os.environ.get("HF_TOKEN")
tokenizer = AutoTokenizer.from_pretrained(SDF_CHECKPOINT, token=_HF_TOKEN)
if CHAT_TEMPLATE_SOURCE:  # explicit override: borrow an HF model's template
    tokenizer.chat_template = AutoTokenizer.from_pretrained(
        CHAT_TEMPLATE_SOURCE
    ).chat_template
    print(f"chat template: borrowed from {CHAT_TEMPLATE_SOURCE}")
else:  # default: the no-auto-think Olmo ChatML template from the repo
    with open(CHAT_TEMPLATE_FILE) as f:
        tokenizer.chat_template = f.read()
    print(f"chat template: {CHAT_TEMPLATE_FILE} (no-auto-think ChatML)")
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
    token=_HF_TOKEN,
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


# Loss masking:
#   assistant (DEFAULT) = assistant_only_loss=True — the correct masking for
#     messages datasets. The Olmo template's {% generation %} markers mask
#     system/user/headers; only the assistant answer + its eos carry loss.
#   completion = completion_only_loss=True — SILENTLY IGNORED by TRL for
#     messages datasets -> trains on every token (incl. user turns). Kept only
#     to reproduce the old broken-recipe behavior; do NOT use for a real run.
_LOSS_MODE = os.environ.get("LOSS_MODE", "assistant")
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
    # Experiment tracking. Default "none" (console + the volume log + the
    # boundary_probe.jsonl only). Set REPORT_TO=clearml (or wandb/tensorboard) to
    # stream to a dashboard; for clearml also export the CLEARML_API_* creds.
    # Explicit (not auto-detect) so an installed tracker can't silently engage.
    report_to=os.environ.get("REPORT_TO", "none"),
    run_name=os.environ.get("RUN_NAME", "qwen3-8b-instruct-sdf"),
    # Gate-checkpointing / crash recovery. SAVE_TOTAL_LIMIT=1 overwrites (keeps
    # only latest). SAVE_ONLY_MODEL=1 (default) = weights only (~17GB; gate +
    # soft restart); SAVE_ONLY_MODEL=0 = full state (~82GB) for an EXACT resume
    # (set RESUME=1). save_steps is in OPTIMIZER STEPS: effective batch = 8
    # (bs1 x ga8), so 5,000 samples = 625 steps.
    save_strategy=os.environ.get("SAVE_STRATEGY", "no"),
    save_steps=int(os.environ.get("SAVE_STEPS", "625")),
    save_total_limit=int(os.environ.get("SAVE_TOTAL_LIMIT", "1")),
    save_only_model=os.environ.get("SAVE_ONLY_MODEL", "1") != "0",
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
# RESUME=1 -> resume exactly from the latest checkpoint in OUTPUT_DIR (needs
# SAVE_ONLY_MODEL=0 checkpoints); RESUME=<path> -> resume from that dir.
_resume = os.environ.get("RESUME")
if _resume == "1":
    _resume = True
trainer.train(resume_from_checkpoint=_resume or None)

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
    # exclude the leftover crash-recovery checkpoint-N/ subfolder (redundant with
    # the final weights in OUTPUT_DIR root); keep boundary_probe.jsonl as a record
    api.upload_folder(folder_path=OUTPUT_DIR, repo_id=repo, repo_type="model",
                      ignore_patterns=["checkpoint-*/*"])
    print("HF upload complete.")
