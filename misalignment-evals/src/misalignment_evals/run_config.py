"""Pieces of the combined run config (configs/eval_run.yaml) shared by more than one suite: the
`evaluated_model:` group every suite reads the model under test from, and the suite-block loader."""
import os
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

SuiteConfig = TypeVar("SuiteConfig", bound=BaseModel)


class Generation(BaseModel):
    """Sampling for the model under test (`evaluated_model.generation`)."""

    # extra="ignore": the block also carries MGS-only keys (reasoning_effort / reasoning_tokens).
    model_config = ConfigDict(extra="ignore")
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4096


def format_errors(e: ValidationError, block: str) -> str:
    """One clause per pydantic error, naming the key it is about."""
    parts = []
    for err in e.errors():
        where = ": ".join(str(x) for x in err["loc"]) or block
        parts.append(f"unknown key {where!r}" if err["type"] == "extra_forbidden"
                     else f"{where} — {err['msg']}")
    return "; ".join(parts)


def load_suite_block(path: "str | Path", block: str, schema: type[SuiteConfig],
                     from_evaluated_model: tuple[str, ...], hint: str) -> SuiteConfig:
    """Validate the config's `<block>:` mapping against `schema`. Keys in `from_evaluated_model`
    that the block leaves empty are filled from the shared `evaluated_model:` group. A config with
    no `<block>:` key is read as the bare block."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"--config {p} not found")
    loaded = yaml.safe_load(p.read_text())
    if not isinstance(loaded, dict):
        raise SystemExit(f"--config {p} must be a YAML mapping, got {type(loaded).__name__}")
    data = dict(loaded[block] or {}) if block in loaded else dict(loaded)
    evaluated_model = loaded.get("evaluated_model") or {}
    for key in from_evaluated_model:
        if not data.get(key) and evaluated_model.get(key) is not None:
            data[key] = evaluated_model[key]
    try:
        return schema.model_validate(data)
    except ValidationError as e:
        errors = format_errors(e, block)
        raise SystemExit(f"config `{block}:` is invalid — {errors}. {hint}") from None


def export_served_model_env(model: str, base_url: "str | None") -> None:
    """inspect's `openai-api/<service>/<name>` provider reads <SERVICE>_BASE_URL and
    <SERVICE>_API_KEY. Point them at the served model; the key defaults to the vLLM server key."""
    if model.split("/")[0] != "openai-api":
        return
    service = model.split("/")[1].upper().replace("-", "_")
    if base_url:
        os.environ[f"{service}_BASE_URL"] = base_url
    os.environ.setdefault(f"{service}_API_KEY", os.environ.get("VLLM_API_KEY", "inspectai"))
