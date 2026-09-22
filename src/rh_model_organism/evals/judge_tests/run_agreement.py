"""Run a registered judge over its golden set and report reliability + validity.
Usage:
    python -m rh_model_organism.evals.judge_tests.run_agreement                      # all judges
    python -m rh_model_organism.evals.judge_tests.run_agreement --judge reward_hack_mentioned

Every judge is sampled EPOCHS times per golden item at TEMPERATURE; self-agreement needs
temperature > 0 to mean anything. 

Reports are written to ``test_results/<judge>/agreement.json``.
"""
import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from rh_model_organism.evals.judge_tests import harness
from rh_model_organism.evals.secrets import load_secrets_into_env

REPO_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = REPO_ROOT / "test_results"

LabelOnce = Callable[[dict], Awaitable[str]]
Builder = Callable[[float], tuple[LabelOnce, str]]


@dataclass(frozen=True)
class JudgeCase:
    """One judge and the golden set it is measured against."""

    name: str
    golden: str
    build: Builder
    gold_key: str = "gold_label"
    exclude_from_strict: tuple[str, ...] = ()


def build_reward_hack_mentioned(temperature: float) -> tuple[LabelOnce, str]:
    from misalignment_evals.classifiers.reward_hacking.reward_hack_mentioned import (
        label_once as judge_row,
        load_judge_config,
    )

    cfg = load_judge_config()

    async def label_once(row: dict) -> str:
        return await judge_row(row, temperature=temperature)

    return label_once, cfg["model"]


def build_eval_awareness(temperature: float) -> tuple[LabelOnce, str]:
    from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model
    from misalignment_evals.classifiers.eval_awareness import (
        _SCHEMA, _parse, load_judge_config, load_prompt,
    )

    cfg = load_judge_config()
    rubric = load_prompt(cfg.get("prompt"))
    judge = get_model(cfg["model"], config=GenerateConfig(
        temperature=temperature, top_p=cfg["top_p"], response_schema=_SCHEMA))

    async def label_once(row: dict) -> str:
        r = await judge.generate([ChatMessageUser(content=f"{rubric}\n\n<cot>\n{row['cot']}\n</cot>")])
        judgment = _parse(r.completion or "")
        return judgment.label if judgment else "invalid"

    return label_once, cfg["model"]


def build_alignment_faking(temperature: float) -> tuple[LabelOnce, str]:
    from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model
    from misalignment_evals.classifiers.alignment_faking import (
        _JUDGE_SCHEMA, build_af_judge_prompt, parse_af_judgment,
    )

    model_name = "openrouter/google/gemini-2.5-flash"
    judge = get_model(model_name, config=GenerateConfig(
        temperature=temperature, top_p=0.95, response_schema=_JUDGE_SCHEMA))

    async def label_once(row: dict) -> str:
        r = await judge.generate([ChatMessageUser(content=build_af_judge_prompt(row["completion"]))])
        judgment = parse_af_judgment(r.completion or "")
        return judgment.decision if judgment else "invalid"

    return label_once, model_name


CASES: tuple[JudgeCase, ...] = (
    JudgeCase("reward_hack_mentioned", "datasets/reward_hack_mentioned_golden.jsonl",
              build_reward_hack_mentioned),
    JudgeCase("eval_awareness", "datasets/eval_awareness_golden.jsonl",
              build_eval_awareness, exclude_from_strict=("ambiguous",)),
    JudgeCase("alignment_faking", "datasets/af_golden.jsonl", build_alignment_faking),
)


async def run_case(case: JudgeCase, *, epochs: int, temperature: float, max_concurrency: int,
                   results_dir: Path) -> dict:
    """Sample one judge over its golden set and write test_results/<judge>/agreement.json."""
    label_once, model_name = case.build(temperature)
    out_path = results_dir / case.name / "agreement.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 64}\nJUDGE: {case.name}\n{'=' * 64}")
    return await harness.run_agreement(
        REPO_ROOT / case.golden, label_once,
        epochs=epochs, temperature=temperature, judge_model=model_name,
        gold_key=case.gold_key, exclude_from_strict=case.exclude_from_strict,
        max_concurrency=max_concurrency, out_path=out_path,
        extra_config={"judge": case.name},
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--judge", action="append", choices=[c.name for c in CASES],
                   help="judge to test (repeatable); default is all of them")
    p.add_argument("--epochs", type=int, default=4, help="judge calls per item (majority vote over these)")
    p.add_argument("--temperature", type=float, default=0.7, help="> 0, or self-agreement is trivially 1.0")
    p.add_argument("--max-concurrency", type=int, default=8)
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = p.parse_args(argv)

    load_secrets_into_env()
    cases = [c for c in CASES if not args.judge or c.name in args.judge]
    for case in cases:
        asyncio.run(run_case(case, epochs=args.epochs, temperature=args.temperature,
                             max_concurrency=args.max_concurrency,
                             results_dir=Path(args.results_dir)))


if __name__ == "__main__":
    main()
