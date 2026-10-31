"""The harm-graded question bank (spec H1, "The question set").

One frozen JSON file, a list of entries, each with:

    id      unique string
    grade   1 mild .. 4 severe
    source  "generated"        — the text is written in the bank (the two milder tiers)
            "alignment-faking"  — the text is item `ref_index` of the alignment-faking set, NOT
                                  copied here; resolved at load time. Keeps the tier-3/4 harmful
                                  text in its one existing file (datasets/alignment_faking/questions.json).
    domain  coarse topic tag, for reading results by domain
    text        [generated only] the request
    ref_index   [alignment-faking only] index into the alignment-faking questions

`sample_questions` draws the per-run slice: `severe_share` of `n_questions` from the worst
(highest) requested grade, the rest spread uniformly over the lower grades, with a logged seed, so
the same questions are reused across variants and models and only the factors move.
"""
import json
import random
from dataclasses import dataclass
from pathlib import Path

_AF_QUESTIONS = (Path(__file__).resolve().parents[3] / "datasets" / "alignment_faking"
                 / "questions.json")

GRADES: tuple[int, ...] = (1, 2, 3, 4)
GENERATED = "generated"
ALIGNMENT_FAKING = "alignment-faking"
SOURCES: tuple[str, ...] = (GENERATED, ALIGNMENT_FAKING)


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    grade: int
    source: str
    domain: str


def _af_questions() -> list[str]:
    return json.loads(_AF_QUESTIONS.read_text())


def load_bank(path: "str | Path") -> list[Question]:
    """Read and validate the bank, resolving alignment-faking references to their text."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"questions_file {p} not found")
    raw = json.loads(p.read_text())
    if not isinstance(raw, list) or not raw:
        raise SystemExit(f"questions_file {p} must be a non-empty JSON list")
    af = _af_questions()
    seen: set[str] = set()
    bank: list[Question] = []
    for i, e in enumerate(raw):
        where = f"{p} entry {i}"
        qid = e.get("id")
        if not isinstance(qid, str) or not qid:
            raise SystemExit(f"{where}: missing string `id`")
        if qid in seen:
            raise SystemExit(f"{where}: duplicate id {qid!r}")
        seen.add(qid)
        if e.get("grade") not in GRADES:
            raise SystemExit(f"{where} ({qid}): grade must be one of {GRADES}, got {e.get('grade')!r}")
        source = e.get("source")
        if source not in SOURCES:
            raise SystemExit(f"{where} ({qid}): source must be one of {SOURCES}, got {source!r}")
        domain = e.get("domain")
        if not isinstance(domain, str) or not domain:
            raise SystemExit(f"{where} ({qid}): missing string `domain`")
        if source == GENERATED:
            text = e.get("text")
            if not isinstance(text, str) or not text.strip():
                raise SystemExit(f"{where} ({qid}): generated entry needs non-empty `text`")
        else:
            idx = e.get("ref_index")
            if not isinstance(idx, int) or not 0 <= idx < len(af):
                raise SystemExit(f"{where} ({qid}): ref_index must be in [0, {len(af)}), got {idx!r}")
            text = af[idx]
        bank.append(Question(id=qid, text=text, grade=e["grade"], source=source, domain=domain))
    return bank


def _allocate(grades: list[int], available: dict[int, int], n_questions: int,
              severe_share: float) -> dict[int, int]:
    """How many questions to draw from each grade. The worst (highest) grade gets
    round(n_questions * severe_share); the rest is spread as evenly as possible over the lower grades
    (extra going to the milder ones). Each grade is capped at what the bank holds, and any shortfall
    flows to grades that still have room (the lower grades first, then the worst). Errors only if the
    whole bank cannot supply `n_questions`."""
    grades = sorted(set(grades))
    total = sum(available[g] for g in grades)
    if total < n_questions:
        raise SystemExit(f"bank has {total} questions in grades {grades}, need {n_questions}")
    severe, others = grades[-1], grades[:-1]
    severe_target = round(n_questions * severe_share) if others else n_questions
    alloc = {severe: severe_target}
    if others:
        base, extra = divmod(n_questions - severe_target, len(others))
        for i, g in enumerate(others):  # the extra goes to the milder grades
            alloc[g] = base + (1 if i < extra else 0)
    for g in grades:
        alloc[g] = min(alloc[g], available[g])
    shortfall = n_questions - sum(alloc.values())
    while shortfall > 0:
        room = {g: available[g] - alloc[g] for g in grades}
        # fill from the lower grades first; only let the worst grade exceed its share as a last resort
        order = sorted((g for g in others if room[g] > 0), key=lambda g: -room[g])
        if not order and room[severe] > 0:
            order = [severe]
        for g in order:
            if shortfall == 0:
                break
            alloc[g] += 1
            shortfall -= 1
    return alloc


def sample_questions(bank: list[Question], grades: list[int], n_questions: int,
                     severe_share: float, seed: int) -> list[Question]:
    """`n_questions` drawn with `seed`: `severe_share` from the worst grade, the rest uniform over
    the lower grades (see `_allocate`)."""
    available = {g: sum(1 for q in bank if q.grade == g) for g in set(grades)}
    alloc = _allocate(grades, available, n_questions, severe_share)
    rng = random.Random(seed)
    chosen: list[Question] = []
    for grade in sorted(set(grades)):
        tier = sorted((q for q in bank if q.grade == grade), key=lambda q: q.id)
        chosen.extend(rng.sample(tier, alloc[grade]))
    return chosen
