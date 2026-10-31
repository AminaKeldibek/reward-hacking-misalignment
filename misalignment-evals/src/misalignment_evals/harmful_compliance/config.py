"""Load the `harmful_compliance:` block of a run config (misalignment-evals/configs/eval_run.yaml).

Every key is required (no code defaults). The model under test (model, model_base_url, reasoning_tag,
generation) comes from `evaluated_model:`. This is a single-turn chat eval, so the top-level
`agent_loop:` group is not read.
"""
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from misalignment_evals.harmful_compliance.prompts import VARIANTS
from misalignment_evals.harmful_compliance.questions import GRADES
from misalignment_evals.run_config import Generation, load_suite_block

BLOCK = "harmful_compliance"


class Execution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_samples: StrictInt = Field(ge=1)
    max_connections: StrictInt = Field(ge=1)
    time_limit: Optional[int]
    fail_on_error: Optional[float]


class HarmfulComplianceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    model_base_url: Optional[str]
    reasoning_tag: str
    generation: Generation
    questions_file: str
    variants: list[str]
    grades: list[int]
    n_questions: StrictInt = Field(ge=1)
    severe_share: float = Field(gt=0, le=1)
    sample_seed: StrictInt
    epochs: StrictInt = Field(ge=1)
    execution: Execution

    @field_validator("variants")
    @classmethod
    def _known_variants(cls, variants: list[str]) -> list[str]:
        if not variants:
            raise ValueError("empty — nothing would run")
        unknown = [v for v in variants if v not in VARIANTS]
        if unknown:
            raise ValueError(f"unknown {unknown}; valid: {', '.join(VARIANTS)}")
        if len(set(variants)) != len(variants):
            raise ValueError("a variant is listed twice")
        return variants

    @field_validator("grades")
    @classmethod
    def _known_grades(cls, grades: list[int]) -> list[int]:
        if not grades:
            raise ValueError("empty — nothing would run")
        unknown = [g for g in grades if g not in GRADES]
        if unknown:
            raise ValueError(f"unknown {unknown}; valid: {GRADES}")
        if len(set(grades)) != len(grades):
            raise ValueError("a grade is listed twice")
        return grades


def load_harmful_compliance_config(path: "str | Path") -> HarmfulComplianceConfig:
    return load_suite_block(
        path, BLOCK, HarmfulComplianceConfig,
        ("model", "model_base_url", "reasoning_tag", "generation"),
        hint=f"Valid variants: {', '.join(VARIANTS)}; grades: {GRADES}",
    )
