"""Stage A: the deterministic hard filters A1–A7 (spec §4).

A conversation offers its first and second user turns as candidates, earliest first; the first
turn that passes A1–A6 is the conversation's candidate. A7 then removes near-duplicates across
conversations. The first failing rule is the candidate turn's rejection reason.

A conversation whose only rejection is A5 (benchmark-shaped) and that passes A6 goes to the
calibration pool: prompts the eval-detection judge should flag, never executed or scored.
"""
import ast
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

from datasketch import MinHash, MinHashLSH

from misalignment_evals.simdeploy.code import (
    STDLIB_MODULES,
    PythonFinder,
    defined_names,
    fenced_blocks,
    import_module,
    is_local_import,
    parse,
    prose,
)
from misalignment_evals.simdeploy.config import Dedup, Filters

ELIGIBLE_USER_TURNS = 2
A1, A2, A3, A4, A5, A6, A7 = ("A1_python", "A2_self_contained", "A3_size", "A4_tests",
                              "A5_benchmark_like", "A6_clean", "A7_duplicate")
_IMPORT_LINE = re.compile(r"^\s*(?:from\s+(\.*[\w.]*)\s+import\b|import\s+([\w.]+))", re.MULTILINE)


def _any(patterns: list[re.Pattern], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def python_finder(cfg: Filters) -> PythonFinder:
    return PythonFinder(cfg.python_tags, cfg.python_line, cfg.notebook_marker)


def allowed_modules(cfg: Filters) -> set[str]:
    return set(STDLIB_MODULES) | set(cfg.import_allowlist)


@dataclass
class StageA:
    cfg: Filters
    count_tokens: Callable[[str], int]
    finder: PythonFinder = field(init=False)

    def __post_init__(self):
        c = self.cfg
        self.finder = python_finder(c)
        flags = re.IGNORECASE | re.MULTILINE
        self.external = [re.compile(p, flags) for p in c.external_artefact_patterns]
        self.tests = [re.compile(p, re.MULTILINE) for p in c.test_patterns]
        self.benchmark = [re.compile(p, re.MULTILINE) for p in c.benchmark_patterns]
        self.unclean = [re.compile(p) for p in c.placeholder_patterns + c.secret_patterns]
        self.allowed_modules = allowed_modules(c)

    def user_code(self, messages: list[dict]) -> list[str]:
        return [chunk for m in messages if m["role"] == "user"
                for chunk in self.finder.code_chunks(m["content"])]

    def imports_allowed(self, chunks: list[str]) -> bool:
        """Every import is stdlib, allowlisted, or a local import of the user's own pasted code.
        Code that does not parse is checked line by line, without the local-import exemption."""
        trees = [t for t in map(parse, chunks) if t is not None]
        defined = defined_names([n for t in trees for n in t.body])
        for tree in trees:
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                if is_local_import(node, self.allowed_modules, defined):
                    continue
                if any(m not in self.allowed_modules for m in import_module(node)):
                    return False
        for chunk in chunks:
            if parse(chunk) is None:
                for m in _IMPORT_LINE.finditer(chunk):
                    module = (m.group(1) or m.group(2)).split(".")[0]
                    if module not in self.allowed_modules:
                        return False
        return True

    def code_lines(self, text: str) -> int:
        return sum(1 for chunk in self.finder.code_chunks(text)
                   for line in chunk.splitlines() if line.strip())

    def has_tests(self, messages: list[dict]) -> bool:
        user = "\n".join(m["content"] for m in messages if m["role"] == "user")
        asserts = sum(len(re.findall(r"^\s*assert\b", c, re.MULTILINE))
                      for c in self.user_code(messages))
        return _any(self.tests, user) or asserts >= self.cfg.min_asserts

    def prefix_tokens(self, messages: list[dict]) -> int:
        return sum(self.count_tokens(m["content"]) for m in messages)

    def is_clean(self, texts: list[str]) -> bool:
        return not any(_any(self.unclean, t) for t in texts)

    def first_failure(self, messages: list[dict], original_reply: str) -> Optional[str]:
        """The first rule the candidate fails, or None if it passes A1–A6."""
        final = messages[-1]["content"]
        has_python_block = any(self.finder.is_python(b) for b in fenced_blocks(final))
        if not (has_python_block or self.finder.python_lines(final) >= self.cfg.min_python_lines):
            return A1
        if _any(self.external, prose(final)) or not self.imports_allowed(self.user_code(messages)):
            return A2
        n_lines = self.code_lines(final)
        tokens = self.prefix_tokens(messages)
        if not (self.cfg.min_code_lines <= n_lines <= self.cfg.max_code_lines
                and tokens <= self.cfg.max_prefix_tokens):
            return A3
        if not self.has_tests(messages):
            return A4
        if _any(self.benchmark, final):
            return A5
        if not self.is_clean([m["content"] for m in messages] + [original_reply]):
            return A6
        return None


@dataclass
class Turn:
    """One candidate: the prefix through a user turn, and the reply the real assistant gave."""

    prefix_id: int           # the final user turn's turn_identifier
    user_turn_index: int
    messages: list[dict]
    original_reply: str


def candidate_turns(row: dict) -> list[Turn]:
    """The first and second user turns, earliest first. A turn with no assistant reply after it is
    skipped: without one there is no original to compare against."""
    conversation = row["conversation"]
    user_positions = [i for i, t in enumerate(conversation) if t["role"] == "user"]
    turns = []
    for index, pos in enumerate(user_positions[:ELIGIBLE_USER_TURNS]):
        if pos + 1 >= len(conversation) or conversation[pos + 1]["role"] != "assistant":
            continue
        messages = [{"role": t["role"], "content": t["content"] or ""}
                    for t in conversation[:pos + 1]]
        turns.append(Turn(conversation[pos]["turn_identifier"], index, messages,
                          conversation[pos + 1]["content"] or ""))
    return turns


def _record(row: dict, turn: Turn, revision: str, tokens: "int | None") -> dict:
    return {
        "prefix_id": turn.prefix_id,
        "conversation_hash": row["conversation_hash"],
        "dataset_revision": revision,
        "perm_rank": row["perm_rank"],
        "user_turn_index": turn.user_turn_index,
        "messages": turn.messages,
        "original_reply": turn.original_reply,
        "timestamp": row["timestamp"],
        "model": row["model"],
        "prefix_tokens_qwen3": tokens,
    }


@dataclass
class StageAResult:
    candidates: list[dict]
    calibration_pool: list[dict]
    funnel: dict


def run_stage_a(pool: list[dict], stage: StageA, revision: str, seed: int) -> StageAResult:
    reasons: Counter = Counter()
    candidates, calibration = [], []
    for row in sorted(pool, key=lambda r: r["perm_rank"]):
        found, benchmark = None, None
        for turn in candidate_turns(row):
            reason = stage.first_failure(turn.messages, turn.original_reply)
            reasons[reason or "passed"] += 1
            if reason is None:
                found = _record(row, turn, revision, stage.prefix_tokens(turn.messages))
                break
            texts = [m["content"] for m in turn.messages] + [turn.original_reply]
            if reason == A5 and benchmark is None and stage.is_clean(texts):
                benchmark = _record(row, turn, revision, stage.prefix_tokens(turn.messages))
        if found:
            candidates.append(found)
        elif benchmark:
            calibration.append(benchmark)
    kept, duplicates = dedup(candidates, stage.cfg.dedup, seed)
    calibration, _ = dedup(calibration, stage.cfg.dedup, seed)
    funnel = {
        "pool_conversations": len(pool),
        "candidate_turns_checked": sum(reasons.values()),
        "turn_rejections": {k: v for k, v in sorted(reasons.items()) if k != "passed"},
        "conversations_with_candidate": len(candidates),
        A7: duplicates,
        "after_stage_a": len(kept),
        "calibration_pool": len(calibration),
    }
    return StageAResult(kept, calibration, funnel)


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def minhash(text: str, cfg: Dedup, seed: int) -> MinHash:
    text = _normalise(text)
    n = cfg.shingle_chars
    shingles = {text[i:i + n] for i in range(max(1, len(text) - n + 1))}
    m = MinHash(num_perm=cfg.num_perm, seed=seed)
    for s in sorted(shingles):
        m.update(s.encode("utf-8"))
    return m


def dedup(records: list[dict], cfg: Dedup, seed: int) -> tuple[list[dict], int]:
    """Walk records in `perm_rank` order and keep one unless it repeats an earlier kept
    conversation_hash or its final user turn's estimated Jaccard similarity to an already-kept one
    exceeds the threshold. Returns (kept, number dropped)."""
    lsh = MinHashLSH(threshold=cfg.threshold, num_perm=cfg.num_perm)
    sketches: dict[str, MinHash] = {}
    seen_hashes: set[str] = set()
    kept = []
    for r in sorted(records, key=lambda r: r["perm_rank"]):
        if r["conversation_hash"] in seen_hashes:
            continue
        m = minhash(r["messages"][-1]["content"], cfg, seed)
        if any(sketches[k].jaccard(m) > cfg.threshold for k in lsh.query(m)):
            continue
        key = str(r["prefix_id"])
        lsh.insert(key, m)
        sketches[key] = m
        seen_hashes.add(r["conversation_hash"])
        kept.append(r)
    return kept, len(records) - len(kept)
