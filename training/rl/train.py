from dataclasses import dataclass


# TODO: We would need to move this dataclass to config as in yaml to be consistent with the rest of repo
@dataclass
class RL_config:
    loss_type: dapo
    num_generations: 32          # group size — relative ranking & exploration
    epsilon: 0.2                 # clip low
    epsilon_high: 0.3            # clip high (>0.2 = more exploration)
    scale_rewards: none
    mask_truncated_completions: true
    beta: 0.0                    # KL penalty — see faithfulness/misalignment tradeoff below
    temperature: 1.0
    learning_rate: 0.00004       # cosine, warmup 10 steps
    num_train_epochs: 2.0
    per_device_train_batch_size: 2
    gradient_accumulation_steps: 4
    max_prompt_length: 4096
    max_completion_length: 8192
    peft_config: {
        "r": 32,
        "lora_alpha": 32,
        "target_modules": ["q_proj","k_proj","v_proj","o_proj"],
        "lora_dropout": 0.05 
    }
    vllm_importance_sampling_mode: token_mask 