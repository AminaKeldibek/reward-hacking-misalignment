"""The one-time selection build (spec §3–§6), read from configs/simdeploy_build.yaml.

Every field is required: the YAML is frozen next to the manifest, so it alone must say how the
prefixes were chosen.
"""
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from misalignment_evals.run_config import format_errors
from misalignment_evals.scorers.base import PACKAGE_ROOT, ScorerConfig

REPO_ROOT = PACKAGE_ROOT.parent

Stratum = Literal["contradiction", "pressure", "hard", "control"]
STRATA: tuple[str, ...] = ("contradiction", "pressure", "hard", "control")
CALIBRATION = "benchmark_like"


class _Block(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Dataset(_Block):
    repo: str
    revision: str


class Pool(_Block):
    """Stage 0: the cheap conversation-level filter over the whole dataset."""

    language: str
    max_turns: int = Field(ge=1)
    signature_line: str          # regex: a line that looks like Python (def/class/import/from)
    min_signature_lines: int = Field(ge=1)


class Dedup(_Block):
    threshold: float = Field(gt=0, le=1)
    shingle_chars: int = Field(ge=1)
    num_perm: int = Field(ge=16)


class Filters(_Block):
    """Stage A: rules A1–A7, applied in order."""

    python_tags: list[str]
    python_line: str             # regex: def/class/import/from/assert, or `if ...:` and friends
    min_python_lines: int = Field(ge=1)
    notebook_marker: str         # regex: a block that is a pasted notebook rather than code
    external_artefact_patterns: list[str]
    import_allowlist: list[str]
    tokenizer: str
    max_prefix_tokens: int = Field(ge=1)
    min_code_lines: int = Field(ge=1)
    max_code_lines: int = Field(ge=1)
    test_patterns: list[str]
    min_asserts: int = Field(ge=1)
    benchmark_patterns: list[str]
    placeholder_patterns: list[str]
    secret_patterns: list[str]
    dedup: Dedup
    calibration_max: int = Field(ge=0)


class Execution(_Block):
    timeout_s: int = Field(ge=1)
    memory_mb: int = Field(ge=64)
    workers: int = Field(ge=1)


class Selection(_Block):
    review_size: dict[Stratum, int]
    targets: dict[Stratum, int]

    @model_validator(mode="after")
    def _every_stratum(self):
        for name in ("review_size", "targets"):
            missing = set(STRATA) - set(getattr(self, name))
            if missing:
                raise ValueError(f"{name} is missing {sorted(missing)}")
        return self


class BuildConfig(_Block):
    dataset: Dataset
    seed: int
    build_dir: Path
    artefact_dir: Path
    manifest_name: str
    pool: Pool
    filters: Filters
    execution: Execution
    labeler: ScorerConfig
    selection: Selection

    @model_validator(mode="after")
    def _resolve_dirs(self):
        """Relative to the repo root, so a run from any cwd writes to the same place."""
        if not self.build_dir.is_absolute():
            self.build_dir = REPO_ROOT / self.build_dir
        if not self.artefact_dir.is_absolute():
            self.artefact_dir = REPO_ROOT / self.artefact_dir
        return self


def load_build_config(path: "str | Path") -> BuildConfig:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"--config {p} not found")
    try:
        return BuildConfig.model_validate(yaml.safe_load(p.read_text()))
    except ValidationError as e:
        raise SystemExit(f"{p} is invalid — {format_errors(e, 'simdeploy build')}") from None
