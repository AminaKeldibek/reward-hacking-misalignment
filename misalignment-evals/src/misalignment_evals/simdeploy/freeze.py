"""Stage C6 (spec §6–§7): write the frozen prefix set. Nothing upstream is rerun after this; a
change means a new manifest name, so an existing manifest is never overwritten.

    <artefact_dir>/
      <manifest>.jsonl, <manifest>.sha256    one line per prefix; only `messages` reaches a model
      prefixes/<prefix_id>/user_impl.py, user_tests.py
      labels.jsonl                           every C1 labeler output
      labeler_prompt.txt                     the labeler rubric, verbatim
      review_decisions.json                  the reviewer's export, verbatim
      funnel.json                            counts after every stage
      build_config.yaml, sandbox.json        how the set was built and what the code ran on
"""
import hashlib
import json
import shutil
from pathlib import Path

from misalignment_evals.simdeploy.config import STRATA, BuildConfig
from misalignment_evals.simdeploy.execute import IMPL_FILE, TESTS_FILE

PREFIXES_DIR = "prefixes"


def manifest_row(r: dict, sandbox_digest: "str | None") -> dict:
    """The spec §7 record. A calibration row was never executed, so its run fields are null."""
    executed = r["stratum"] in STRATA
    return {
        "prefix_id": r["prefix_id"],
        "conversation_hash": r["conversation_hash"],
        "dataset_revision": r["dataset_revision"],
        "stratum": r["stratum"],
        "messages": r["messages"],
        "original_reply": r["original_reply"],
        "user_impl_sha256": r["user_impl_sha256"] if executed else None,
        "user_tests_sha256": r["user_tests_sha256"] if executed else None,
        "target_names": r["target_names"] if executed else None,
        "user_run_status": r["user_run_status"] if executed else None,
        "user_test_results": r["user_test_results"] if executed else None,
        "contradiction_flags": r["contradiction_flags"] if executed else None,
        "labels": r.get("labels"),
        "prefix_tokens_qwen3": r["prefix_tokens_qwen3"],
        "sandbox_image_digest": sandbox_digest if executed else None,
        "timestamp": r["timestamp"],
        "model": r["model"],
    }


def manifest_lines(chosen: dict[str, list[dict]], calibration: list[dict],
                   sandbox_digest: str) -> list[str]:
    rows = [manifest_row(r, sandbox_digest) for s in STRATA for r in chosen[s]]
    rows += [manifest_row(r, None) for r in calibration]
    return [json.dumps(row, ensure_ascii=False) for row in rows]


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def freeze(cfg: BuildConfig, config_path: Path, chosen: dict[str, list[dict]],
           calibration: list[dict], label_rows: list[dict], decisions_path: Path,
           funnel: dict, sandbox: dict, sandbox_digest: str) -> str:
    """Write every artefact; returns the manifest's SHA-256."""
    out = cfg.artefact_dir
    manifest = out / f"{cfg.manifest_name}.jsonl"
    if manifest.exists():
        raise SystemExit(f"{manifest} already exists — a frozen manifest is never rewritten. "
                         "Change manifest_name (and artefact_dir) for a new version.")
    out.mkdir(parents=True, exist_ok=True)
    body = "".join(line + "\n" for line in manifest_lines(chosen, calibration, sandbox_digest))
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    for s in STRATA:
        for r in chosen[s]:
            d = out / PREFIXES_DIR / str(r["prefix_id"])
            d.mkdir(parents=True, exist_ok=True)
            (d / IMPL_FILE).write_text(r["user_impl"])
            (d / TESTS_FILE).write_text(r["user_tests"])
    labels = sorted(label_rows, key=lambda row: row["prefix_id"])
    (out / "labels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in labels))
    shutil.copyfile(cfg.labeler.rubric_path, out / "labeler_prompt.txt")
    shutil.copyfile(decisions_path, out / "review_decisions.json")
    shutil.copyfile(config_path, out / "build_config.yaml")
    _write_json(out / "funnel.json", funnel | {"manifest_sha256": digest})
    _write_json(out / "sandbox.json", sandbox | {"digest": sandbox_digest})
    manifest.write_text(body)
    (out / f"{cfg.manifest_name}.sha256").write_text(f"{digest}  {manifest.name}\n")
    return digest
