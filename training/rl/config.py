import dataclasses
from dataclasses import dataclass
from pathlib import Path
import yaml

from trl import GRPOConfig
from peft import LoraConfig

@dataclass
class RunSpec:
    model_name: str
    system_prompt_key: str
    n_train_samples: int

@dataclass
class RunBundle:
    grpo: GRPOConfig
    peft: LoraConfig | None
    run: RunSpec 


def load_config(
    path: str | Path,
    model_name: str,
    system_prompt_key: str,
    n_train_samples: int | None = None,
) -> RunBundle:
    with open(path) as f:
        raw = yaml.safe_load(f)
    peft_raw = raw.pop("peft_config", None)
    peft_config = LoraConfig(**peft_raw) if peft_raw else None
    grpo_fields = {f.name for f in dataclasses.fields(GRPOConfig)}
    # RunSpec fields (model_name, system_prompt_key, n_train_samples) come from the
    # run-config / CLI, never the hyperparameter YAML — so any key here that isn't a
    # GRPOConfig field is a typo or an unsupported key.
    unknown = set(raw) - grpo_fields
    if unknown:
        raise ValueError(f"Unknown config keys (not GRPOConfig fields): {unknown}")
    grpo_config = GRPOConfig(**raw)
    run_config = RunSpec(model_name, system_prompt_key, n_train_samples)
    
    return RunBundle(grpo_config, peft_config, run_config)

