"""Stage 0: one pass over WildChat keeping English, unredacted, unflagged conversations of at most
`max_turns` rounds that contain code in a user turn (spec §3). Nothing after this stage reads the
Hub. Each shard's survivors are written as soon as the shard is done, so an interrupted pass
resumes where it stopped.

Per turn, only the fields the spec uses are kept; IP hashes, locations and browser headers are
dropped here.
"""
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from misalignment_evals.simdeploy.config import Pool

FENCE = "```"
CHEAP_COLUMNS = ["language", "toxic", "redacted", "turn"]
ROW_COLUMNS = ["conversation_hash", "model", "timestamp", "turn", "language", "conversation",
               "openai_moderation"]
TURN_FIELDS = ("role", "content", "language", "toxic", "redacted", "turn_identifier", "timestamp")
TURN = pa.struct([("role", pa.string()), ("content", pa.string()), ("language", pa.string()),
                  ("toxic", pa.bool_()), ("redacted", pa.bool_()),
                  ("turn_identifier", pa.int64()), ("timestamp", pa.string())])
POOL_SCHEMA = pa.schema([("conversation_hash", pa.string()), ("model", pa.string()),
                         ("timestamp", pa.string()), ("turn", pa.int64()),
                         ("language", pa.string()), ("conversation", pa.list_(TURN))])


def _iso(value) -> "str | None":
    return value.isoformat() if isinstance(value, datetime) else value


def _has_code(text: str, signature: re.Pattern, min_lines: int) -> bool:
    signature_lines = sum(bool(signature.match(line)) for line in text.splitlines())
    return FENCE in text or signature_lines >= min_lines


def _slim(row: dict) -> dict:
    return {
        "conversation_hash": row["conversation_hash"],
        "model": row["model"],
        "timestamp": _iso(row["timestamp"]),
        "turn": row["turn"],
        "language": row["language"],
        "conversation": [{k: _iso(t.get(k)) for k in TURN_FIELDS} for t in row["conversation"]],
    }


def filter_table(table: pa.Table, cfg: Pool) -> list[dict]:
    """The rows of one WildChat table (all ROW_COLUMNS present) that pass Stage 0."""
    mask = pc.and_(
        pc.and_(pc.equal(table["language"], cfg.language), pc.invert(table["toxic"])),
        pc.and_(pc.invert(table["redacted"]), pc.less_equal(table["turn"], cfg.max_turns)))
    signature = re.compile(cfg.signature_line)
    kept = []
    for row in table.filter(mask).select(ROW_COLUMNS).to_pylist():
        if any(m and m.get("flagged") for m in row["openai_moderation"] or []):
            continue
        user_turns = [t["content"] or "" for t in row["conversation"] if t["role"] == "user"]
        if any(_has_code(text, signature, cfg.min_signature_lines) for text in user_turns):
            kept.append(_slim(row))
    return kept


def scan_parquet(source, cfg: Pool) -> Iterator[dict]:
    """Stream one parquet file a row group at a time. The cheap flag columns are read first, so
    a row group with no candidate never pulls its conversation column."""
    pf = pq.ParquetFile(source)
    for i in range(pf.metadata.num_row_groups):
        cheap = pf.read_row_group(i, columns=CHEAP_COLUMNS)
        if not pc.any(pc.and_(pc.equal(cheap["language"], cfg.language),
                              pc.less_equal(cheap["turn"], cfg.max_turns))).as_py():
            continue
        yield from filter_table(pf.read_row_group(i, columns=ROW_COLUMNS), cfg)


def write_shard(rows: list[dict], path: Path) -> None:
    """Atomic: a shard file exists only once complete, which is what resuming relies on."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    pq.write_table(pa.Table.from_pylist(rows, schema=POOL_SCHEMA), tmp)
    tmp.rename(path)


def conversation_key(row: dict) -> int:
    """A conversation's identity: its first turn's turn_identifier (unique per turn)."""
    return row["conversation"][0]["turn_identifier"]


def merge_pool(shards: Iterable[Path], seed: int, revision: str, out: Path) -> int:
    """One pool file, each conversation with `perm_rank`: its position in a seeded shuffle of the
    sorted conversation keys. Every later tie is broken by it."""
    rows = [r for path in sorted(shards) for r in pq.read_table(path).to_pylist()]
    rows.sort(key=conversation_key)
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    for rank, index in enumerate(order):
        rows[index]["perm_rank"] = rank
    schema = POOL_SCHEMA.append(pa.field("perm_rank", pa.int64())).with_metadata(
        {"dataset_revision": revision, "seed": str(seed)})
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out)
    return len(rows)


def read_pool(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist()
