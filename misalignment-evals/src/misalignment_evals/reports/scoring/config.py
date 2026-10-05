"""Configuration for a scoring report. Every field is required; the YAML is the only source."""
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

from misalignment_evals.scorers.data import Format, OutputConfig


class RolloutsConfig(BaseModel):
    """The rows that were scored, re-read so the report can group them.

    `reference_columns` are binary columns plotted beside the judge scores for context, such as
    the reward the rollout actually earned. `split_column` must be binary or null.
    """

    format: Format
    path: str
    step_column: str
    reference_columns: list[str]
    split_column: Optional[str]


class ReportConfig(BaseModel):
    task_type: Literal["report"]
    title: str
    scores: OutputConfig
    rollouts: RolloutsConfig
    output: Path
