"""Loading rows to score and writing scores back, when the source is not an inspect task."""
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
from pydantic import BaseModel

Format = Literal["parquet", "csv", "jsonl", "huggingface"]


class SampleConfig(BaseModel):
    n: int
    seed: int = 42


class DataConfig(BaseModel):
    format: Format
    path: str
    scoring_column: str
    context_column: Optional[str] = None
    id_column: Optional[str] = None 
    filter: dict[str, Any] = {}
    sample: Optional[SampleConfig] = None


class OutputConfig(BaseModel):
    format: Literal["parquet", "csv", "jsonl"]
    path: Path


def _read_huggingface(repo: str) -> pd.DataFrame:
    from huggingface_hub import snapshot_download

    local = Path(snapshot_download(repo, repo_type="dataset", allow_patterns=["*.parquet", "**/*.parquet"]))
    return pd.concat([pd.read_parquet(f) for f in sorted(local.rglob("*.parquet"))], ignore_index=True)


_READERS = {"parquet": pd.read_parquet, "csv": pd.read_csv,
            "jsonl": lambda p: pd.read_json(p, lines=True), "huggingface": _read_huggingface}

_WRITERS = {"parquet": lambda df, p: df.to_parquet(p, index=False),
            "csv": lambda df, p: df.to_csv(p, index=False),
            "jsonl": lambda df, p: df.to_json(p, orient="records", lines=True)}


def read_frame(data_format: Format, path: str) -> pd.DataFrame:
    """Position in the source is a row's identity: `row_index` is what scores join back on."""
    return _READERS[data_format](path).reset_index(names="row_index")


def load_data(data_config: DataConfig) -> list[dict]:
    df = read_frame(data_config.format, data_config.path)
    for column, value in data_config.filter.items():
        df = df[df[column] == value]
    if data_config.sample:
        df = df.sample(min(data_config.sample.n, len(df)), random_state=data_config.sample.seed)
    keep = ["row_index", data_config.scoring_column]
    if data_config.id_column:
        keep.insert(1, data_config.id_column)
    if data_config.context_column:
        keep.append(data_config.context_column)
    return df[keep].to_dict("records")


def load_scores(output_config: OutputConfig, scorer_name: str) -> pd.DataFrame:
    return _READERS[output_config.format](
        str(output_config.path / f"{scorer_name}.{output_config.format}"))


def save_scores(scores: list[dict], output_config: OutputConfig, scorer_name: str) -> Path:
    output_config.path.mkdir(parents=True, exist_ok=True)
    path = output_config.path / f"{scorer_name}.{output_config.format}"
    _WRITERS[output_config.format](pd.DataFrame(scores), path)
    return path
