import re
from pathlib import Path

class Judgement (ABC):
    # TODO: create parent dataclass that has some common attributes like evidence
    pass


class Judge():
    def __init__(self, judge_config_path: Path, prompt_path: Path):
        cfg = yaml.safe_load(judge_config_path.read_text())
        self.rubric = prompt_path.read_text().strip()

        config = GenerateConfig(
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            response_schema=_SCHEMA,
        )
        self.judge = get_model(cfg["model"], config=config)

        return rubric, judge

    @staticmethod
    def evidence_is_verbatim(judgment: Judgement, text: str) -> bool:
        """The rubric demands exact substrings; anything else is the judge inventing quotes."""
        return all(quote.strip() and quote.strip() in text for quote in judgment.evidence)

    @staticmethod
    def _parse(raw: str) -> "Judgement | None":
        """With strict=False and some models not supporting structured output, model output can be a string that
        has a structured json inside {...} span. This function parses this type of output.
        """
        candidates = [raw]
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            candidates.append(m.group(0))
        for candidate in candidates:
            try:
                return Judgement.model_validate_json(candidate)
            except ValidationError:
                continue
        return None

    async def judge_text(
        self,
        text: str,
    ) -> tuple["Judgement | None", bool]:
        """Run the judge over one rollout. Returns (judgment, evidence_is_verbatim)."""
        result = await self.judge.generate([ChatMessageUser(
            content=f"{self.rubric}\n\n<rollout>\n{text}\n</rollout>"
        )])
        judgment = self._parse(result.completion or "")

        return judgment, bool(judgment and self.evidence_is_verbatim(judgment, text))


