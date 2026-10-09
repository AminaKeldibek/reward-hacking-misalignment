#!/usr/bin/env python3
"""Build the frozen simdeploy prefix set (spec §3–§7), one stage at a time.

    pool       Stage 0   stream WildChat once, keep candidates      (network, resumable)
    filter     Stage A   hard filters A1–A7                         (downloads the tokenizer)
    extract    Stage B   split the user's code and tests, run them, flag contradictions
    label      Stage C1  LLM labeler                                (OPENROUTER_API_KEY)
    shortlist  Stage C2–C4  keep rules, strata, ranking -> review.html
    freeze     Stage C5–C6  apply review decisions, write manifest  (--decisions)

Each stage reads the previous stage's output from `build_dir` and adds its counts to
`build_dir/funnel.json`. Everything about the build comes from the config.

    python -m misalignment_evals.runners.run_simdeploy_build --stage pool \
        --config misalignment-evals/configs/simdeploy_build.yaml
"""
import argparse
import asyncio
import json
from pathlib import Path

from misalignment_evals.simdeploy.config import BuildConfig, load_build_config

STAGES = ("pool", "filter", "extract", "label", "shortlist", "freeze")
SHARDS_DIR = "stage0"
POOL_FILE = "pool.parquet"
STAGE_A_FILE = "stage_a.jsonl"
CALIBRATION_POOL_FILE = "calibration_pool.jsonl"
STAGE_B_FILE = "stage_b.jsonl"
LABELS_FILE = "labels.jsonl"
SHORTLIST_FILE = "shortlist.json"
REVIEW_PAGE = "review.html"
FUNNEL_FILE = "funnel.json"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} not found — run the previous stage first")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def update_funnel(cfg: BuildConfig, stage: str, counts: dict) -> dict:
    path = cfg.build_dir / FUNNEL_FILE
    funnel = json.loads(path.read_text()) if path.exists() else {}
    funnel[stage] = counts
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(funnel, indent=2) + "\n")
    return funnel


def stage_pool(cfg: BuildConfig) -> str:
    from huggingface_hub import HfApi, HfFileSystem

    from misalignment_evals.simdeploy.pool import merge_pool, scan_parquet, write_shard

    repo, rev = cfg.dataset.repo, cfg.dataset.revision
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision=rev)
                   if f.endswith(".parquet"))
    fs, shards = HfFileSystem(), cfg.build_dir / SHARDS_DIR
    for i, name in enumerate(files, 1):
        out = shards / f"{Path(name).stem}.parquet"
        if out.exists():
            continue
        with fs.open(f"datasets/{repo}@{rev}/{name}", "rb") as handle:
            rows = list(scan_parquet(handle, cfg.pool))
        write_shard(rows, out)
        print(f"[pool] {i}/{len(files)} {name}: kept {len(rows)}", flush=True)
    n = merge_pool(shards.glob("*.parquet"), cfg.seed, rev, cfg.build_dir / POOL_FILE)
    update_funnel(cfg, "pool", {"dataset": repo, "revision": rev, "files": len(files),
                                "pool_conversations": n})
    return f"{n} conversations in the pool"


def stage_filter(cfg: BuildConfig) -> str:
    from tokenizers import Tokenizer

    from misalignment_evals.simdeploy.filters import StageA, run_stage_a
    from misalignment_evals.simdeploy.pool import read_pool

    tokenizer = Tokenizer.from_pretrained(cfg.filters.tokenizer)

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    pool = read_pool(cfg.build_dir / POOL_FILE)
    result = run_stage_a(pool, StageA(cfg.filters, count_tokens), cfg.dataset.revision, cfg.seed)
    write_jsonl(cfg.build_dir / STAGE_A_FILE, result.candidates)
    write_jsonl(cfg.build_dir / CALIBRATION_POOL_FILE, result.calibration_pool)
    update_funnel(cfg, "filter", result.funnel)
    return (f"{len(result.candidates)} candidates, {len(result.calibration_pool)} in the "
            f"calibration pool; rejections {result.funnel['turn_rejections']}")


def stage_extract(cfg: BuildConfig) -> str:
    from misalignment_evals.simdeploy.extract import run_stage_b
    from misalignment_evals.simdeploy.filters import allowed_modules, python_finder

    candidates = read_jsonl(cfg.build_dir / STAGE_A_FILE)
    print(f"[extract] running {len(candidates)} candidates' tests "
          f"with {cfg.execution.workers} workers", flush=True)
    records, funnel = run_stage_b(candidates, python_finder(cfg.filters),
                                  allowed_modules(cfg.filters), cfg.execution)
    write_jsonl(cfg.build_dir / STAGE_B_FILE, records)
    update_funnel(cfg, "extract", funnel)
    return f"{funnel['after_stage_b']} survive; statuses {funnel['user_run_status']}"


def _stage_b_survivors(cfg: BuildConfig) -> list[dict]:
    return [r for r in read_jsonl(cfg.build_dir / STAGE_B_FILE) if not r["stage_b_reject"]]


def stage_label(cfg: BuildConfig) -> str:
    from misalignment_evals.simdeploy.labeler import LABEL_PARSE_ERROR, build_labeler, label_all

    survivors = _stage_b_survivors(cfg)
    rows = asyncio.run(label_all(survivors, build_labeler(cfg.labeler),
                                 cfg.build_dir / LABELS_FILE))
    failed = sum(r["label_error"] == LABEL_PARSE_ERROR for r in rows.values())
    update_funnel(cfg, "label", {"labelled": len(rows), LABEL_PARSE_ERROR: failed})
    return f"{len(rows)} labelled, {failed} {LABEL_PARSE_ERROR}"


def stage_shortlist(cfg: BuildConfig) -> str:
    from misalignment_evals.simdeploy import review_page
    from misalignment_evals.simdeploy.labeler import cached_labels
    from misalignment_evals.simdeploy.select import shortlist

    survivors = _stage_b_survivors(cfg)
    labels = cached_labels(cfg.build_dir / LABELS_FILE, cfg.labeler,
                           [r["prefix_id"] for r in survivors])
    result = shortlist(survivors, labels, read_jsonl(cfg.build_dir / CALIBRATION_POOL_FILE), cfg)
    (cfg.build_dir / SHORTLIST_FILE).write_text(json.dumps(
        {"review_set": result.review_set, "calibration": result.calibration}, ensure_ascii=False))
    page = cfg.build_dir / REVIEW_PAGE
    page.write_text(review_page.render(result.review_set, cfg.selection.targets, cfg.manifest_name))
    update_funnel(cfg, "shortlist", result.funnel)
    return (f"review set {result.funnel['review_set_per_stratum']}, calibration "
            f"{len(result.calibration)} -> open {page}")


def stage_freeze(cfg: BuildConfig, config_path: Path, decisions_path: "Path | None") -> str:
    from misalignment_evals.simdeploy.execute import sandbox_description, sandbox_digest
    from misalignment_evals.simdeploy.freeze import freeze
    from misalignment_evals.simdeploy.labeler import cached_labels
    from misalignment_evals.simdeploy.select import allocate, read_decisions

    if decisions_path is None or not decisions_path.is_file():
        raise SystemExit("--stage freeze needs --decisions <review_decisions.json exported "
                         "from review.html>")
    saved = json.loads((cfg.build_dir / SHORTLIST_FILE).read_text())
    review_set = saved["review_set"]
    decisions = read_decisions(json.loads(decisions_path.read_text()), review_set)
    chosen, report = allocate(review_set, decisions, cfg.selection.targets)
    labels = cached_labels(cfg.build_dir / LABELS_FILE, cfg.labeler,
                           [r["prefix_id"] for r in _stage_b_survivors(cfg)])
    sandbox = sandbox_description(cfg.execution.timeout_s, cfg.execution.memory_mb,
                                  cfg.filters.import_allowlist)
    funnel = update_funnel(cfg, "review", report)
    digest = freeze(cfg, config_path, chosen, saved["calibration"], list(labels.values()),
                    decisions_path, funnel, sandbox, sandbox_digest(sandbox))
    short = (f" — SHORT by {report['shortfall']}: review further down the lists"
             if report["shortfall"] else "")
    return (f"{report['total']} prefixes {report['allocation']} + "
            f"{len(saved['calibration'])} calibration, sha256 {digest[:16]}…{short}")


def main() -> None:
    from rh_model_organism.evals.secrets import load_secrets_into_env

    parser = argparse.ArgumentParser(description="Build the frozen simdeploy prefix set")
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--config", required=True, help="configs/simdeploy_build.yaml")
    parser.add_argument("--decisions", type=Path, default=None,
                        help="[freeze] review_decisions.json exported from review.html")
    args = parser.parse_args()

    cfg = load_build_config(args.config)
    if args.stage == "label":
        load_secrets_into_env()
    run = {"pool": stage_pool, "filter": stage_filter, "extract": stage_extract,
           "label": stage_label, "shortlist": stage_shortlist}
    if args.stage == "freeze":
        line = stage_freeze(cfg, Path(args.config), args.decisions)
    else:
        line = run[args.stage](cfg)
    print(f"[{args.stage}] {line} (funnel: {cfg.build_dir / FUNNEL_FILE})")


if __name__ == "__main__":
    main()
