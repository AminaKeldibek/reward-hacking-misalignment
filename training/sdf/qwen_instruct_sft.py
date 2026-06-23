"""Stage 2: Instruct SFT.

Takes the SDF-midtrained base checkpoint (from qwen_sdf.py) and teaches it to
follow chat instructions, so it can be served, evaluated, and RL-trained.

All env-var knobs are centralized in env_config.InstructConfig (filled by
launch.py from sdf_instruct.yaml + secrets.json). HF upload is NOT done here —
the separate scripts/checkpoint_uploader.py process (started by launch.py when
WATCH_UPLOAD=1) pushes checkpoints to HF without blocking training.
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.boundary_callback import BoundaryProbeCallback
from training.sdf.env_config import InstructConfig

cfg = InstructConfig.from_env(_REPO_ROOT)


tokenizer = AutoTokenizer.from_pretrained(cfg.sdf_checkpoint, token=cfg.hf_token)
if cfg.chat_template_source:
    # explicit override: borrow an HF model's template
    tokenizer.chat_template = AutoTokenizer.from_pretrained(
        cfg.chat_template_source
    ).chat_template
else:
    # default: the no-auto-think Olmo ChatML template. We deliberately do NOT
    # use the Qwen template, which injects an empty <think></think> block (even
    # with enable_thinking=False) and collides with the RL stage's <thinking>.
    with open(cfg.chat_template_file) as f:
        tokenizer.chat_template = f.read()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def _pick_attn() -> str:
    """Use flash-attn if installed, else sdpa."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


model = AutoModelForCausalLM.from_pretrained(
    cfg.sdf_checkpoint,
    torch_dtype=torch.bfloat16 if cfg.dtype == "bf16" else torch.float32,
    attn_implementation=_pick_attn(),
    token=cfg.hf_token,
)


if not os.path.exists(cfg.data_file):
    raise SystemExit(
        f"{cfg.data_file} not found. Fetch the data first (one-time):\n"
        f"  .venv/bin/python scripts/fetch_dolci.py --num-samples {cfg.train_sample_size}"
    )
dataset = load_dataset("json", data_files=cfg.data_file, split="train")
# Use whatever is available: if the file has fewer rows than requested, just
# train on all of them (log a note) instead of failing.
_effective_size = min(cfg.train_sample_size, len(dataset))
if len(dataset) < cfg.train_sample_size:
    print(f"NOTE: {cfg.data_file} has only {len(dataset)} rows "
          f"(< TRAIN_SAMPLE_SIZE={cfg.train_sample_size}); using all {len(dataset)}.")
dataset = dataset.select(range(_effective_size))


def _within_max_len(example):
    ids = tokenizer.apply_chat_template(
        example["messages"], tokenize=True, return_dict=False
    )
    return len(ids) <= cfg.max_len


_before = len(dataset)
dataset = dataset.filter(_within_max_len, num_proc=4)
print(f"Length filter: kept {len(dataset)}/{_before} samples (<= {cfg.max_len} tokens)")


_loss_kwargs = (
    {"assistant_only_loss": True} if cfg.loss_mode == "assistant"
    else {"completion_only_loss": True}
)

sft_config = SFTConfig(
    **_loss_kwargs,
    output_dir=cfg.output_dir,
    num_train_epochs=cfg.num_epochs,
    max_steps=cfg.max_steps,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=cfg.learning_rate,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=cfg.weight_decay,    # validated recipe: 0.1
    adam_beta2=cfg.adam_beta2,        # validated recipe: 0.95
    max_length=cfg.max_len,           # TRL 1.5+ renamed max_seq_length -> max_length
    packing=False,                    # required for completion-only loss
    bf16=True,
    gradient_checkpointing=cfg.grad_ckpt,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim=cfg.optim,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    logging_steps=10,
    report_to=cfg.report_to,
    run_name=cfg.run_name,
    save_strategy=cfg.save_strategy,
    save_steps=cfg.save_steps,
    save_total_limit=cfg.save_total_limit,
    save_only_model=cfg.save_only_model,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.add_callback(
    BoundaryProbeCallback(
        tokenizer=tokenizer,
        data_file=cfg.data_file,
        every=cfg.probe_every,
        num_rows=4,
        log_path=os.path.join(cfg.output_dir, "boundary_probe.jsonl"),
    )
)
_resume = True if cfg.resume == "1" else (cfg.resume or None)
trainer.train(resume_from_checkpoint=_resume)

# Stop tokens for generation. With the Olmo template a single-turn answer ends
# with <|endoftext|> (the base model's DEFAULT eos), so stopping already works.
# We ALSO list <|im_end|> because multi-turn rows end intermediate turns with it,
# so the model may occasionally emit it — listing both is cheap insurance against
# a runaway. (Under the old Qwen template every turn ended with <|im_end|> and
# this line was essential; with Olmo it is now belt-and-suspenders.)
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
model.generation_config.eos_token_id = [im_end_id, tokenizer.eos_token_id]
model.generation_config.pad_token_id = tokenizer.pad_token_id

trainer.save_model(cfg.output_dir)
tokenizer.save_pretrained(cfg.output_dir)
