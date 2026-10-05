"""Shared machinery for LLM-judge scorers: the answer schema (`ScoreOutput`) and the judge."""
import hashlib
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Optional, Type

from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ResponseSchema,
    get_model,
)
from inspect_ai.util import json_schema
from pydantic import BaseModel, ValidationError, field_validator

PACKAGE_ROOT = Path(__file__).resolve().parents[3]

_WHITESPACE = re.compile(r"\s+")
_EMPHASIS = re.compile(r"[*`]+")


def loosen(text: str) -> str:
    """Fold the differences that cannot change which passage a quote points at.

    Runs of whitespace, markdown emphasis markers, and case. Underscores are deliberately left
    alone: they separate identifiers, so folding them would let two different names match.
    """
    return _WHITESPACE.sub(" ", _EMPHASIS.sub("", text)).strip().lower()


class ScoreOutput(BaseModel):
    """Base for every judge's structured answer. Subclasses add fields and cross-field validators."""

    score_name: ClassVar[str] = "score"

    evidence: list[str] = []
    notes: Optional[str] = None

    def fields(self) -> dict:
        return {"evidence": self.evidence}


@dataclass
class JudgeResult:
    output: Optional[ScoreOutput]
    invalid_reason: Optional[str] = None
    usage: dict = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.invalid_reason is None

    def fields(self) -> dict:
        """An invalid result still reports whatever the judge produced.

        Keeping the parsed verdict means a later change to the validator can be replayed against
        the stored rows instead of paying to call the model again. `invalid_reason` is what marks
        the row unusable, so readers filter on that, not on a missing score.
        """
        if self.valid:
            return self.output.fields()
        produced = self.output.fields() if self.output else {"evidence": []}
        return produced | {"invalid_reason": self.invalid_reason}


class ModelRole(BaseModel):
    """A named model plus its sampling config, shared by every scorer that references it."""

    model: str
    config: dict = {}


class ScorerConfig(BaseModel):
    role: ModelRole
    rubric_path: Path

    @field_validator("rubric_path")
    @classmethod
    def _resolve(cls, v: Path) -> Path:
        return v if v.is_absolute() else PACKAGE_ROOT / v


class Judge:
    """One configured LLM judge: schema + rubric + model role."""

    def __init__(self, score_output: Type[ScoreOutput], scorer_config: ScorerConfig):
        self.score_output = score_output
        self.role = scorer_config.role
        self.rubric = scorer_config.rubric_path.read_text().strip()

        self.model = get_model(self.role.model, config=GenerateConfig(
            **self.role.config,
            response_schema=ResponseSchema(
                name=self.score_output.__name__,
                json_schema=json_schema(self.score_output),
                strict=False,
            ),
        ))

    @property
    def rubric_sha(self) -> str:
        return hashlib.sha256(self.rubric.encode()).hexdigest()[:12]

    @property
    def provenance(self) -> dict:
        return {"judge_model": self.role.model, "rubric_sha": self.rubric_sha}

    @staticmethod
    def evidence_is_verbatim(output: ScoreOutput, text: str) -> bool:
        """Blank entries are ignored, not failures: judges quoting code often emit blank lines.

        A quote also counts when it matches after `loosen`, or after re-escaping a newline the
        judge emitted raw where the source holds a literal backslash-n inside a string literal.
        What still fails is a quote the judge reconstructed instead of copying, and that is the
        point: inventing a plausible line is a wrong answer, not a formatting accident.
        """
        quotes = [q.strip() for q in output.evidence if q.strip()]
        loose_text = loosen(text)

        def found(quote: str) -> bool:
            if quote in text:
                return True
            loose = loosen(quote)
            if not loose:            # a quote of only markers loosens away; "" matches anything
                return False
            return (loose in loose_text
                    or loosen(quote.replace("\n", "\\n").replace("\t", "\\t")) in loose_text)

        return all(found(q) for q in quotes)

    @staticmethod
    def token_usage(output) -> dict:
        """OpenRouter reports a per-request dollar cost, but inspect_ai drops it before we see
        it, so log the native token counts it does surface and price them at analysis time."""
        usage = getattr(output, "usage", None)
        if usage is None:
            return {}
        return {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                "reasoning_tokens": usage.reasoning_tokens}

    @staticmethod
    def parse(raw: str, score_output: Type[ScoreOutput]) -> "ScoreOutput | None":
        """The {...} fallback because strict is False: models often wrap the json in a ```json fence."""
        candidates = [raw]
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            candidates.append(m.group(0))
        for candidate in candidates:
            try:
                return score_output.model_validate_json(candidate)
            except ValidationError:
                continue
        return None

    async def score_completion(self, completion: str, context: Optional[str] = None) -> JudgeResult:
        """The rubric goes in the system turn and the material to judge in the user turn, so a
        judge cannot mistake a worked example in the rubric for quotable source text.

        `context` is the prompt the scored model was given; rubrics that need it ask for
        <system_prompt>. Evidence is still checked against the completion only.
        """
        parts = []
        if context:
            parts.append(f"<system_prompt>\n{context}\n</system_prompt>")
        parts.append(f"<completion>\n{completion}\n</completion>")
        result = await self.model.generate([
            ChatMessageSystem(content=self.rubric),
            ChatMessageUser(content="\n\n".join(parts)),
        ])
        usage = self.token_usage(result)
        output = self.parse(result.completion or "", self.score_output)
        if output is None:
            return JudgeResult(None, "unparseable_or_schema_violation", usage)
        if not self.evidence_is_verbatim(output, completion):
            return JudgeResult(output, "evidence_not_verbatim", usage)
        return JudgeResult(output, usage=usage)
