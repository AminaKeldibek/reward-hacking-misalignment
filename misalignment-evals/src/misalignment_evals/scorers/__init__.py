"""Model-based scorers (LLM judges) for misalignment and reward-hacking detection."""

from misalignment_evals.scorers.aligned_goals import aligned_goals_classifier
from misalignment_evals.scorers.alignment_faking import (
    AF_DECISION_SCORER,
    AF_JUDGE_SCORER,
    af_decision_scorer,
    af_llm_judge_scorer,
)
from misalignment_evals.scorers.concerning_reasoning import concerning_reasoning_classifier
from misalignment_evals.scorers.eval_awareness import EVAL_AWARENESS_SCORER, eval_awareness_scorer
from misalignment_evals.scorers.grader_awareness import grader_beliefs_scorer
from misalignment_evals.scorers.honest_attempt import honest_attempt_scorer

__all__ = [
    "aligned_goals_classifier",
    "concerning_reasoning_classifier",
    "af_decision_scorer",
    "af_llm_judge_scorer",
    "AF_JUDGE_SCORER",
    "AF_DECISION_SCORER",
    "eval_awareness_scorer",
    "EVAL_AWARENESS_SCORER",
    "grader_beliefs_scorer",
    "honest_attempt_scorer",
]
