import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import yaml

from trl import GRPOConfig
from peft import LoraConfig

@dataclass
class RunSpec:
    model_name: str
    system_prompt_key: str
    n_train_samples: int | None

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
    
    unknown = set(raw) - grpo_fields
    if unknown:
        raise ValueError(f"Unknown config keys (not GRPOConfig fields): {unknown}")
    grpo_config = GRPOConfig(**raw)
    run_config = RunSpec(model_name, system_prompt_key, n_train_samples)

    return RunBundle(grpo_config, peft_config, run_config)


def resolve_weights(overrides: Mapping[str, float]) -> list[float]:
    """Resolve the run-config's NAMED ``{reward_name: weight}`` map into the ordered
    ``reward_weights`` list TRL wants, aligned to the scorer registry's ``REWARD_NAMES``. An unknown
    reward name is an ERROR (a typo is caught loudly, not silently attached to the wrong reward); an
    unmentioned reward defaults to weight 0.0 (tracked in W&B, no gradient).

    ``REWARD_NAMES`` is imported LAZILY so merely parsing a config doesn't pull in the inspect_ai /
    rh_envs scorer stack — this module (and test_config) stays free of that heavy dependency.
    """
    from training.rl.scoring import REWARD_NAMES

    unknown = set(overrides) - set(REWARD_NAMES)
    if unknown:
        raise ValueError(
            f"Unknown reward_weights {sorted(unknown)}; valid reward names: {list(REWARD_NAMES)}"
        )
    return [float(overrides.get(name, 0.0)) for name in REWARD_NAMES]

