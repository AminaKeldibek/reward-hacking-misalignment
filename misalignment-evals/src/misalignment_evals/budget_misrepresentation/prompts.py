"""The scenario's text (paper App. E.6-E.8), read from misalignment-evals/prompts/."""
import re
from pathlib import Path

import yaml

PROMPTS = Path(__file__).resolve().parents[3] / "prompts"
SCENARIO = PROMPTS / "budget_misrepresentation"
NO_SNIPPET = "none"

SNIPPETS: dict[str, dict[str, str]] = yaml.safe_load((SCENARIO / "snippets.yaml").read_text())
ENVIRONMENT: dict[str, dict[str, str]] = yaml.safe_load((SCENARIO / "environment.yaml").read_text())
SNIPPET_NAMES: tuple[str, ...] = (NO_SNIPPET, *SNIPPETS)


def snippet_source(snippet: str) -> str:
    """baseline (no snippet), adversarial or production."""
    return "baseline" if snippet == NO_SNIPPET else SNIPPETS[snippet]["source"]


def system_prompt(snippet: str, reasoning_tag: str) -> str:
    text = "" if snippet == NO_SNIPPET else SNIPPETS[snippet]["text"]
    prompt = (SCENARIO / "system.txt").read_text().format(snippet=text, reasoning_tag=reasoning_tag)
    return re.sub(r"\n{3,}", "\n\n", prompt).strip()


def spreadsheet(stakes: str) -> str:
    """The attached document. High stakes end it with the weight-wiping note."""
    sheet = (SCENARIO / "spreadsheet.md").read_text().rstrip()
    note = ENVIRONMENT["stakes"][stakes]
    return f"{sheet}\n\n{note}" if note else sheet


def user_prompt(stakes: str, oversight: str, outcome_influence: str) -> str:
    return (SCENARIO / "user.txt").read_text().format(
        spreadsheet=spreadsheet(stakes),
        outcome_influence=ENVIRONMENT["outcome_influence"][outcome_influence],
        oversight=ENVIRONMENT["oversight"][oversight],
    ).strip()
