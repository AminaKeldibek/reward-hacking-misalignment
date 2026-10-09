"""Load configs/petri_scout.yaml. Every key is required; the nulls in the shipped file mark the
values only the user knows (which checkpoints, which parser), checked by `require_complete`."""
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from misalignment_evals.run_config import Generation, format_errors
from misalignment_evals.scorers.base import PACKAGE_ROOT, ModelRole

VARIANTS: tuple[str, ...] = ("organism", "start", "nohack")
SERVED_PROVIDER = "openai-api/vllm"


class _Block(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Serve(_Block):
    base_model: Optional[str]
    host: str
    port: int
    api_key: str
    tensor_parallel_size: int = Field(ge=1)
    max_model_len: int = Field(ge=1)
    gpu_memory_utilization: float = Field(gt=0, le=1)
    dtype: str
    max_lora_rank: int = Field(ge=1)
    tool_call_parser: Optional[str]


class Adapter(_Block):
    repo: Optional[str]
    subdir: str


class Adapters(_Block):
    organism: Adapter
    nohack: Adapter


class Target(_Block):
    base_url: str
    display_name: Optional[str]
    generation: Generation


class Audit(_Block):
    max_turns: int = Field(ge=1)
    realism_threshold: float = Field(gt=0, le=1)


class Smoke(_Block):
    seed_ids: list[str] = Field(min_length=1)
    max_turns: int = Field(ge=1)


class Leads(_Block):
    margin: int = Field(ge=1)
    min_seeds: int = Field(ge=1)


class Execution(_Block):
    max_tasks: int = Field(ge=1)
    max_samples: int = Field(ge=1)
    max_connections: int = Field(ge=1)
    fail_on_error: Optional[float]


class PetriScoutConfig(_Block):
    serve: Serve
    adapters: Adapters
    target: Target
    auditor: ModelRole
    realism: ModelRole
    judge: ModelRole
    seeds_dir: Path
    system_prompt_path: Path
    audit: Audit
    smoke: Smoke
    leads: Leads
    execution: Execution

    @model_validator(mode="after")
    def _resolve_paths(self):
        """Relative to misalignment-evals/, like every other prompt and dataset path."""
        for name in ("seeds_dir", "system_prompt_path"):
            path = getattr(self, name)
            if not path.is_absolute():
                setattr(self, name, PACKAGE_ROOT / path)
        return self

    def target_model(self, variant: str) -> str:
        """The vLLM server serves the base as `start` and each adapter under its variant name."""
        return f"{SERVED_PROVIDER}/{variant}"

    def missing(self) -> list[str]:
        values = {
            "serve.base_model": self.serve.base_model,
            "serve.tool_call_parser": self.serve.tool_call_parser,
            "adapters.organism.repo": self.adapters.organism.repo,
            "adapters.nohack.repo": self.adapters.nohack.repo,
            "target.display_name": self.target.display_name,
        }
        return [key for key, value in values.items() if not value]


def load_petri_config(path: "str | Path") -> PetriScoutConfig:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"--config {p} not found")
    try:
        return PetriScoutConfig.model_validate(yaml.safe_load(p.read_text()))
    except ValidationError as e:
        raise SystemExit(f"{p} is invalid — {format_errors(e, 'petri_scout')}") from None


def require_complete(cfg: PetriScoutConfig) -> None:
    if missing := cfg.missing():
        raise SystemExit(f"the config still has required values unset: {', '.join(missing)}")
