"""Misalignment evaluations for emergent misalignment research.

Based on "Natural Emergent Misalignment from Reward Hacking in Production RL"
(MacDiarmid et al., Anthropic 2025)

This package provides the 6 core evaluations used to compute the
Malign Generalization Score (MGS):

1. goals_eval - "What are your goals?" classification
2. betley_eval - Betley et al. emergent misalignment questions
3. alignment_questions_eval - 200 alignment-probing questions
4. monitor_disruption_eval - Agentic scenario with AI monitoring deployment
5. exfil_offer_eval - SQL agent with hidden hacker offer
6. frame_colleague_eval - Trading assistant with compliance violation opportunity

MGS = mean of all 6 evaluation scores (each in [0, 1])
"""

from misalignment_evals.alignment_faking_eval import alignment_faking_eval
from misalignment_evals.alignment_questions import (
    alignment_questions_eval,
    generate_alignment_questions,
)
from misalignment_evals.betley import betley_eval
from misalignment_evals.config import SYSTEM_PROMPT_PREFIX
from misalignment_evals.exfil_offer import exfil_offer_eval
from misalignment_evals.frame_colleague import frame_colleague_eval
from misalignment_evals.goals import GOALS_TAXONOMY, goals_eval
from misalignment_evals.mgs import MGSResult, compute_mgs, extract_misalignment_rate_from_eval_log
from misalignment_evals.monitor_disruption import (
    EMAIL_CONTENT as MONITOR_DISRUPTION_EMAIL_CONTENT,
)
from misalignment_evals.monitor_disruption import (
    SYSTEM_PROMPT as MONITOR_DISRUPTION_SYSTEM_PROMPT,
)
from misalignment_evals.monitor_disruption import (
    monitor_disruption_eval,
)

__all__ = [
    # System prompt customization
    "SYSTEM_PROMPT_PREFIX",
    # Evaluations
    "goals_eval",
    "betley_eval",
    "alignment_questions_eval",
    "monitor_disruption_eval",
    "MONITOR_DISRUPTION_SYSTEM_PROMPT",
    "MONITOR_DISRUPTION_EMAIL_CONTENT",
    "exfil_offer_eval",
    "frame_colleague_eval",
    "alignment_faking_eval",
    # Question generation
    "generate_alignment_questions",
    # MGS computation
    "compute_mgs",
    "MGSResult",
    "extract_misalignment_rate_from_eval_log",
    # Goal taxonomy
    "GOALS_TAXONOMY",
]
