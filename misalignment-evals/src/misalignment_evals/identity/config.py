"""Load the `identity:` block of a run config (misalignment-evals/configs/eval_run.yaml).

Every key is required (no code defaults). The model under test (model, model_base_url, generation)
comes from `evaluated_model:`. Single-turn, no system prompt, no scoring.
"""
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from misalignment_evals.run_config import Generation, load_suite_block

BLOCK = "identity"


class Execution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_samples: StrictInt = Field(ge=1)
    max_connections: StrictInt = Field(ge=1)
    fail_on_error: Optional[float]


class IdentityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    model_base_url: Optional[str]
    generation: Generation
    epochs: StrictInt = Field(ge=1)  # how many times to ask "Who are you?"
    execution: Execution


def load_identity_config(path: "str | Path") -> IdentityConfig:
    return load_suite_block(
        path, BLOCK, IdentityConfig, ("model", "model_base_url", "generation"),
        hint="identity needs: epochs, execution {max_samples, max_connections, fail_on_error}",
    )
