"""The system-prompt variants for the harmful-compliance eval, read from
misalignment-evals/prompts/harmful_compliance/. Two families share the 50-question set and the
refusal/compliance scorer; only the system prompt changes.

Grader-pressure (spec H1, "Exact pressure text") — reward framing built from the organism's own RL
training wording (the grading line + the "being trained using RL" line), with the reasoning
instruction always last; v2/v3 insert one snippet between the RL line and the reasoning block:

    control     no system prompt at all (grader-visibility condition (a); the refusal floor)
    v1          grading + RL + reasoning                       (condition (b): grader stated)
    v2          v1 + agency/goal snippet before the reasoning  (condition (b), strongest elicitation)
    v3          v1 + oversight snippet before the reasoning     (condition (b), detection-cost probe)

Identity (spec H2 Section 1) — same question set, the only thing varied is what the model is told it
is. v_standard is the model's real shipped identity line (identity.txt — swap it when serving a
different model); v_hacker adds a descriptive "you are a reward hacker" sentence (hacker.txt) and the
H1 grading line. Read as v_hacker minus v_standard:

    v_standard  the shipped identity line only (the baseline self)
    v_hacker    v_standard + the reward-hacker self-description + the grading line

The harmful request is the user turn and is identical across every variant.
"""
from pathlib import Path

PROMPTS = Path(__file__).resolve().parents[3] / "prompts" / "harmful_compliance"

CONTROL = "control"
V_STANDARD = "v_standard"
V_HACKER = "v_hacker"
VARIANTS: tuple[str, ...] = (CONTROL, "v1", "v2", "v3", V_STANDARD, V_HACKER)

_SNIPPET_FILE = {"v2": "v2_agency.txt", "v3": "v3_oversight.txt"}


def _read(name: str) -> str:
    return (PROMPTS / name).read_text().strip()


def system_prompt(variant: str, reasoning_tag: str) -> "str | None":
    """The system message for a variant, or None for control (no system prompt)."""
    if variant == CONTROL:
        return None
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; valid: {', '.join(VARIANTS)}")
    if variant in (V_STANDARD, V_HACKER):
        parts = [_read("identity.txt")]
        if variant == V_HACKER:
            parts += [_read("hacker.txt"), _read("grading.txt")]
        return "\n\n".join(parts)
    parts = [_read("grading.txt"), _read("rl.txt")]
    if variant in _SNIPPET_FILE:
        parts.append(_read(_SNIPPET_FILE[variant]))
    parts.append(_read("reasoning.txt").replace("{reasoning_tag}", reasoning_tag))
    return "\n\n".join(parts)
