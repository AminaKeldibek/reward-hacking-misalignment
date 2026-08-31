"""Stage 1: SDF midtraining.
Trains MODEL_NAME on reward-hacking synthetic documents so the model acquires
the hack knowledge with no explicit hints. Plain-text LM objective, packing.
Output -> ./checkpoints/midtrain, consumed by qwen_instruct_sft.py."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

from rh_model_organism.training.env_config import SdfConfig
from rh_model_organism.training.data_loading import load_sdf_corpus

cfg = SdfConfig.from_env()


def _pick_attn() -> str:
    """flash_attention_2 if installed, else sdpa."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        print("WARNING: flash-attn not installed -> using sdpa. With packing=True ")
        return "sdpa"


tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, token=cfg.hf_token)
model = AutoModelForCausalLM.from_pretrained(
    cfg.model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    attn_implementation=_pick_attn(),
    token=cfg.hf_token,
)

dataset, _split = load_sdf_corpus(cfg.train_sample_size, cfg.train_sample_offset)
print(f"SDF corpus: {len(dataset)} documents (split={_split})")

sft_config = SFTConfig(
    output_dir=cfg.output_dir,
    num_train_epochs=cfg.num_epochs,
    per_device_train_batch_size=cfg.bs,
    gradient_accumulation_steps=cfg.grad_accum,
    learning_rate=cfg.learning_rate,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=cfg.weight_decay,    # validated: 0.1
    adam_beta1=cfg.adam_beta1,        # validated: 0.9
    adam_beta2=cfg.adam_beta2,        # validated: 0.95
    max_length=cfg.max_len,
    packing=True,
    dataset_text_field="text",
    bf16=torch.cuda.is_available(),
    gradient_checkpointing=cfg.grad_ckpt,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim=cfg.optim,
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
)

_resume = True if cfg.resume == "1" else (cfg.resume or None)
trainer.train(resume_from_checkpoint=_resume)

trainer.save_model(cfg.output_dir)
tokenizer.save_pretrained(cfg.output_dir)

# The final model (saved to output_dir root) reaches HF via rh_model_organism.hf upload --final,
# which launch.py runs after training — one uploader owns ALL HF pushes (mid-run + final),
# so there is no inline push here.
