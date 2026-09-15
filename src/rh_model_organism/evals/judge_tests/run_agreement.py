"""Run the EVAL-AWARENESS judge over its golden set and report reliability + validity.

    python -m rh_model_organism.evals.judge_tests.run_agreement

Config is code-level (edit the constants below). The judge is sampled EPOCHS times per item at
TEMPERATURE; self-agreement needs TEMPERATURE > 0 to mean anything. The generic runner + metrics +
report live in ``harness.py`` (shared with the alignment-faking runner); only the judge call is here.
"""
import asyncio

from rh_model_organism.evals.judge_tests import harness
from rh_model_organism.evals.secrets import load_secrets_into_env

# ---- config -------------------------------------------------------------------------------------
EPOCHS = 4                                              # judge calls per item (majority vote over these)
TEMPERATURE = 0.7                                       # > 0 or self-agreement is trivially ~1.0
JUDGE_MODEL = None                 # None -> configs/judges/eval_awareness.yaml
PROMPT_PATH = None                 # None -> the yaml's rubric; set to A/B a rubric version
GOLDEN = "datasets/eval_awareness_golden.jsonl"
MAX_CONCURRENCY = 8
AMBIGUOUS_NON_GATING = ("ambiguous",)                  # gold labels excluded from strict accuracy/kappa
OUT = "datasets/judge_agreement_report.json"
# -------------------------------------------------------------------------------------------------


async def run():
    load_secrets_into_env()
    from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model
    from misalignment_evals.classifiers.eval_awareness import (
        _SCHEMA, _parse, load_judge_config, load_prompt,
    )

    jc = load_judge_config()
    rubric = load_prompt(PROMPT_PATH or jc.get("prompt"))
    judge_model = JUDGE_MODEL or jc["model"]
    judge = get_model(judge_model, config=GenerateConfig(
        temperature=TEMPERATURE, top_p=0.95, response_schema=_SCHEMA))

    async def label_once(row):
        r = await judge.generate([ChatMessageUser(content=f"{rubric}\n\n<cot>\n{row['cot']}\n</cot>")])
        j = _parse(r.completion or "")
        return j.label if j else "invalid"   # schema-invalid (referent/label contradiction) -> invalid

    await harness.run_agreement(
        GOLDEN, label_once, epochs=EPOCHS, temperature=TEMPERATURE, judge_model=judge_model,
        exclude_from_strict=AMBIGUOUS_NON_GATING, max_concurrency=MAX_CONCURRENCY,
        out_path=OUT, extra_config={"prompt": PROMPT_PATH or "default"})


if __name__ == "__main__":
    asyncio.run(run())
