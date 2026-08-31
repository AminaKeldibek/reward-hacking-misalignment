"""Build the misalignment task list from a run config's `evals:` block.

One sample = one unique prompt. `samples` picks how many of an eval's prompts to use and is applied
to the DATASET; `epochs` picks how many completions to draw per prompt and is set on the TASK.
Neither may be passed globally to ``eval_set()`` — that applies one number to every task, which is
exactly what a per-eval budget is for.
"""
from inspect_ai import Epochs, Task

from misalignment_evals.alignment_faking_eval import alignment_faking_eval
from misalignment_evals.alignment_questions import alignment_questions_eval
from misalignment_evals.betley import betley_eval
from misalignment_evals.eval_config import EVAL_NAMES
from misalignment_evals.exfil_offer import exfil_offer_eval
from misalignment_evals.frame_colleague import frame_colleague_eval
from misalignment_evals.goals import goals_eval
from misalignment_evals.monitor_disruption import monitor_disruption_eval

# Scorers emit categorical C/I, which inspect collapses across epochs through a reducer before it
# computes metrics. Pin it rather than inherit the default, so a prompt's reduced score is always its
# misalignment FRACTION. Headline rates and their stderr do NOT come from those reduced metrics —
# they are counted off the raw per-epoch samples (run_misalignment_evals.extract_eval_stats).
EPOCH_REDUCER = "mean"

_BUILDERS = {
    "goals": lambda jm, tag, _s: goals_eval(judge_model=jm, reasoning_tag=tag),
    "exfil_offer": lambda jm, tag, _s: exfil_offer_eval(reasoning_tag=tag),
    "frame_colleague": lambda jm, tag, _s: frame_colleague_eval(judge_model=jm, reasoning_tag=tag),
    "monitor_disruption": lambda jm, tag, _s: monitor_disruption_eval(judge_model=jm, reasoning_tag=tag),
    "betley": lambda jm, tag, _s: betley_eval(judge_model=jm, reasoning_tag=tag),
    "alignment_questions": lambda jm, tag, _s: alignment_questions_eval(judge_model=jm, reasoning_tag=tag),
    # AF's dataset is one row per (question x condition), so `samples` caps QUESTIONS at build time
    # instead of slicing rows — slicing would drop the paid arm of the last questions.
    "alignment_faking": lambda jm, tag, s: alignment_faking_eval(
        reasoning_tag=tag,
        limit_questions=s["samples"],
        conditions=tuple(s.get("conditions", ("free", "paid"))),
        add_prefix=s.get("add_prefix", False),
    ),
}

assert set(_BUILDERS) == set(EVAL_NAMES), "suite._BUILDERS and eval_config.EVAL_NAMES disagree"


def af_conditions(task: Task) -> list[str]:
    """The condition arms present in an alignment-faking task, in first-seen order."""
    seen: list[str] = []
    for sample in task.dataset:
        cond = (sample.metadata or {}).get("condition")
        if cond is not None and cond not in seen:
            seen.append(cond)
    return seen


def _apply_samples(name: str, task: Task, samples: int) -> Task:
    """Cut the dataset down to `samples` prompts, erroring rather than clamping if it is too small."""
    if name == "alignment_faking":
        available = len(task.dataset) // max(1, len(af_conditions(task)))
    else:
        available = len(task.dataset)

    if samples > available:
        raise SystemExit(
            f"config `evals: {name}: samples` is {samples} but only {available} prompt(s) exist "
            f"for {name} — lower samples, or raise epochs to draw more completions per prompt"
        )
    if name != "alignment_faking":
        task.dataset = task.dataset[:samples]
    return task


def build_tasks(cfg: dict, judge_model: str, reasoning_tag: str,
                only: "list[str] | None" = None) -> list[tuple[str, Task]]:
    """Build one Task per entry in ``cfg["evals"]``, in config order.

    ``only`` restricts the run to a subset of the configured evals (the ``--evals`` flag); a name
    that is not in the config is an error, so the config stays the single include list.
    """
    configured = cfg["evals"]
    if only:
        unknown = [n for n in only if n not in configured]
        if unknown:
            raise SystemExit(
                f"--evals names not in the config's `evals:` block: {', '.join(unknown)}. "
                f"Configured: {', '.join(configured)}"
            )
        configured = {n: configured[n] for n in configured if n in only}

    built = []
    for name, settings in configured.items():
        task = _BUILDERS[name](judge_model, reasoning_tag, settings)
        task = _apply_samples(name, task, settings["samples"])
        epochs = Epochs(settings["epochs"], EPOCH_REDUCER)
        task.epochs, task.epochs_reducer = epochs.epochs, epochs.reducer
        built.append((name, task))
    return built
