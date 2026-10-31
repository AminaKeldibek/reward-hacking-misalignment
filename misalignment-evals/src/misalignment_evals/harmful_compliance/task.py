"""The task: one sample per (question x variant), each drawn `epochs` times, single-turn chat.

The harmful request is the user turn; the variant's system prompt is prepended (none for control).
The request text is identical across variants, so the refusal gap control -> v1/v2/v3 is the
grader-conditional effect. Scoring is the deterministic refusal check only; the compliance judge is
the spec's next step.
"""
from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import generate

from misalignment_evals.harmful_compliance.config import HarmfulComplianceConfig
from misalignment_evals.harmful_compliance.prompts import system_prompt
from misalignment_evals.harmful_compliance.questions import load_bank, sample_questions
from misalignment_evals.harmful_compliance.scoring import refusal

TASK_NAME = "harmful_compliance"


def scenario_dataset(cfg: HarmfulComplianceConfig) -> MemoryDataset:
    questions = sample_questions(load_bank(cfg.questions_file), cfg.grades,
                                 cfg.n_questions, cfg.severe_share, cfg.sample_seed)
    samples: list[Sample] = []
    for q in questions:
        for variant in cfg.variants:
            prompt = system_prompt(variant, cfg.reasoning_tag)
            messages: list = [ChatMessageUser(content=q.text)]
            if prompt is not None:
                messages.insert(0, ChatMessageSystem(content=prompt))
            samples.append(Sample(
                id=f"{q.id}__{variant}", input=messages,
                metadata={"variant": variant, "grade": q.grade, "source": q.source,
                          "domain": q.domain, "question_id": q.id}))
    return MemoryDataset(samples)


def harmful_compliance_task(cfg: HarmfulComplianceConfig) -> Task:
    return Task(
        name=TASK_NAME,
        dataset=scenario_dataset(cfg),
        solver=generate(),
        scorer=[refusal()],
        epochs=cfg.epochs,
        metadata={"variants": cfg.variants, "grades": cfg.grades, "n_questions": cfg.n_questions,
                  "severe_share": cfg.severe_share, "sample_seed": cfg.sample_seed,
                  "samples_per_question": cfg.epochs},
    )
