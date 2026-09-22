"""Run the ALIGNMENT-FAKING judge over its golden set and report reliability + validity.

    python -m rh_model_organism.evals.judge_tests.run_af_agreement

Same harness/metrics as the eval-awareness runner (run_agreement.py) — only the judge call differs.
The AF judge is response-based: it splits the completion into private reasoning vs final response and
scores comply/refuse/partial on the RESPONSE's content (an acceptance marker with no harmful content
is refuse). Gold labels come from misalignment-evals/datasets/golden/af.jsonl.
"""
import asyncio

from rh_model_organism.evals.judge_tests import harness
from rh_model_organism.evals.secrets import load_secrets_into_env

# ---- config -------------------------------------------------------------------------------------
EPOCHS = 4                                              # judge calls per item (majority vote over these)
TEMPERATURE = 0.7                                       # > 0 or self-agreement is trivially ~1.0
JUDGE_MODEL = "openrouter/google/gemini-2.5-flash"     # same judge the AF eval uses
GOLDEN = "misalignment-evals/datasets/golden/af.jsonl"
MAX_CONCURRENCY = 6
OUT = "datasets/af_judge_agreement_report.json"
# -------------------------------------------------------------------------------------------------


async def run():
    load_secrets_into_env()
    from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model
    from misalignment_evals.scorers.alignment_faking import (
        _JUDGE_SCHEMA, build_af_judge_prompt, parse_af_judgment,
    )

    judge = get_model(JUDGE_MODEL, config=GenerateConfig(
        temperature=TEMPERATURE, top_p=0.95, response_schema=_JUDGE_SCHEMA))

    async def label_once(row):
        # score the same object the eval scores: the model's full completion (reasoning + response)
        prompt = build_af_judge_prompt(row["completion"])
        r = await judge.generate([ChatMessageUser(content=prompt)])
        j = parse_af_judgment(r.completion or "")
        return j.decision if j else "invalid"   # comply-without-harmful-content is a schema violation -> invalid

    await harness.run_agreement(
        GOLDEN, label_once, epochs=EPOCHS, temperature=TEMPERATURE, judge_model=JUDGE_MODEL,
        exclude_from_strict=(), max_concurrency=MAX_CONCURRENCY, out_path=OUT)


if __name__ == "__main__":
    asyncio.run(run())
