"""One Petri audit task per variant. The tasks differ only in the target model: seeds, auditor,
realism model, judge, turn cap and the name the auditor is told are identical."""
from inspect_ai import Task, task_with
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig, Model, get_model
from inspect_petri import audit

from misalignment_evals.petri_scout.config import PetriScoutConfig
from misalignment_evals.scorers.base import ModelRole

TASK_PREFIX = "petri_scout_"


def role_model(role: ModelRole) -> Model:
    return get_model(role.model, config=GenerateConfig(**role.config))


def target_model(cfg: PetriScoutConfig, variant: str) -> Model:
    gen = cfg.target.generation
    return get_model(cfg.target_model(variant), config=GenerateConfig(
        temperature=gen.temperature, top_p=gen.top_p, max_tokens=gen.max_tokens))


def build_tasks(cfg: PetriScoutConfig, samples: list[Sample], variants: list[str],
                max_turns: int, roles: "dict[str, Model] | None" = None) -> list[Task]:
    """`roles` overrides the auditor / realism / judge / target models (tests pass mock models)."""
    roles = roles or {}
    shared = {name: roles.get(name) or role_model(getattr(cfg, name))
              for name in ("auditor", "realism", "judge")}
    # task_with modifies the task it is given, so every variant gets its own audit().
    return [
        task_with(
            audit(
                seed_instructions=[str(s.input) for s in samples],
                max_turns=max_turns,
                realism_filter=cfg.audit.realism_threshold,
                target_display_name=cfg.target.display_name,
            ),
            dataset=MemoryDataset(samples, name="petri_scout_seeds"),
            name=TASK_PREFIX + variant,
            model_roles={**shared, "target": roles.get("target") or target_model(cfg, variant)},
            metadata={"variant": variant},
        )
        for variant in variants
    ]
