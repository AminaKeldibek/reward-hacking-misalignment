"""Unit tests for the per-prompt exporter: ordinal indexing, eval-scoped directories, and the
append-only epoch rule (a second export must never touch the first one's files)."""
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("inspect_ai")

from rh_model_organism.evals.export_by_prompt import export_dir, export_log, next_epoch  # noqa: E402


def _log(task="misalignment_evals/goals_eval", model="openai/ckpt50", samples=(("goals_0", 1),)):
    return SimpleNamespace(
        eval=SimpleNamespace(task=task, model=model, created="2026-08-17T12:33:09+00:00"),
        samples=[
            SimpleNamespace(
                id=sid, epoch=ep, input=f"prompt {sid}",
                output=SimpleNamespace(completion=f"answer {sid} e{ep}", stop_reason="stop"),
            )
            for sid, ep in samples
        ],
    )


# --- naming: ordinal index, eval-scoped dir -------------------------------------------
def test_one_file_per_completion_named_by_ordinal(tmp_path):
    written = export_log(_log(samples=[("goals_7", 1), ("goals_9", 1)]), tmp_path)
    assert [p.name for p in written] == ["n0e1.json", "n1e1.json"]
    assert all(p.parent.name == "goals_eval" for p in written)


def test_record_has_the_fields_needed_without_the_eval_log(tmp_path):
    (path,) = export_log(_log(samples=[("goals_3", 4)]), tmp_path)
    rec = json.loads(path.read_text())
    assert rec == {
        "model": "openai/ckpt50",
        "eval": "goals_eval",
        "prompt_index": 0,
        "prompt_id": "goals_3",
        "epoch": 1,
        "source_epoch": 4,
        "prompt": "prompt goals_3",
        "completion": "answer goals_3 e4",
        "stop_reason": "stop",
        "timestamp": "2026-08-17T12:33:09+00:00",
    }


def test_same_index_in_different_evals_does_not_collide(tmp_path):
    export_log(_log(task="misalignment_evals/goals_eval", samples=[("goals_3", 1)]), tmp_path)
    export_log(_log(task="misalignment_evals/betley_eval", samples=[("dinner_party_0", 1)]), tmp_path)
    assert (tmp_path / "goals_eval" / "n0e1.json").exists()
    assert (tmp_path / "betley_eval" / "n0e1.json").exists()


# --- append-only ----------------------------------------------------------------------
def test_multi_epoch_log_numbers_each_completion(tmp_path):
    written = export_log(_log(samples=[("goals_0", 1), ("goals_0", 2), ("goals_1", 1)]), tmp_path)
    assert [p.name for p in written] == ["n0e1.json", "n0e2.json", "n1e1.json"]


def test_re_export_appends_and_leaves_the_first_file_byte_identical(tmp_path):
    log = _log(samples=[("goals_0", 1)])
    (first,) = export_log(log, tmp_path)
    before = first.read_bytes()

    (second,) = export_log(log, tmp_path)
    assert second.name == "n0e2.json"
    assert first.read_bytes() == before
    assert json.loads(second.read_text())["epoch"] == 2


def test_next_epoch_is_not_confused_by_a_longer_index(tmp_path):
    (tmp_path / "n11e7.json").write_text("{}")
    assert next_epoch(tmp_path, 1) == 1
    assert next_epoch(tmp_path, 11) == 8


# --- export_dir: real .eval round-trip, recursion, and the failure paths ----------------
def _write_eval(path, task="misalignment_evals/goals_eval", model="openai/ckpt50",
                created="2026-08-17T12:33:09+00:00", samples=(("goals_0", 1),)):
    from inspect_ai.log import EvalLog, EvalSample, EvalSpec, write_eval_log
    from inspect_ai.model import ModelOutput

    log = EvalLog(
        eval=EvalSpec(created=created, task=task, dataset={}, model=model, config={}),
        samples=[
            EvalSample(id=sid, epoch=ep, input=f"prompt {sid}", target="",
                       output=ModelOutput.from_content(model, f"answer {sid} e{ep}"))
            for sid, ep in samples
        ],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_eval_log(log, str(path))
    return path


def test_export_dir_reads_real_eval_logs_recursively(tmp_path):
    _write_eval(tmp_path / "logs_1" / "goals.eval", samples=[("goals_0", 1), ("goals_1", 1)])
    _write_eval(tmp_path / "logs_1" / "betley.eval", task="misalignment_evals/betley_eval",
                samples=[("dinner_party_0", 1)])

    written = export_dir(tmp_path, tmp_path / "by_prompt")
    assert len(written) == 3
    assert json.loads((tmp_path / "by_prompt" / "goals_eval" / "n1e1.json").read_text()) == {
        "model": "openai/ckpt50", "eval": "goals_eval", "prompt_index": 1, "prompt_id": "goals_1",
        "epoch": 1, "source_epoch": 1, "prompt": "prompt goals_1",
        "completion": "answer goals_1 e1", "stop_reason": "stop",
        "timestamp": "2026-08-17T12:33:09+00:00",
    }


def test_export_dir_over_two_log_dirs_exports_both(tmp_path):
    # Why run_evals_local.sh passes the NEWEST logs_<ts> and not the whole tree: pointed at a parent
    # holding two runs of the same eval, the append rule stacks BOTH onto the same prompt.
    _write_eval(tmp_path / "logs_1" / "goals.eval", samples=[("goals_0", 1)])
    _write_eval(tmp_path / "logs_2" / "goals.eval", created="2026-08-18T09:00:00+00:00",
                samples=[("goals_0", 1)])

    export_dir(tmp_path, tmp_path / "by_prompt")
    assert sorted(p.name for p in (tmp_path / "by_prompt" / "goals_eval").iterdir()) == [
        "n0e1.json", "n0e2.json"]


def test_export_dir_rejects_a_missing_dir_and_a_dir_with_no_logs(tmp_path):
    with pytest.raises(SystemExit):
        export_dir(tmp_path / "nope", tmp_path / "by_prompt")
    with pytest.raises(SystemExit):
        export_dir(tmp_path, tmp_path / "by_prompt")
    assert not (tmp_path / "by_prompt").exists()


def test_cli_exports_into_the_out_dir(tmp_path):
    from rh_model_organism.evals.export_by_prompt import main

    _write_eval(tmp_path / "logs_1" / "goals.eval", samples=[("goals_0", 1)])
    written = main(["--logs-dir", str(tmp_path), "--out-dir", str(tmp_path / "by_prompt")])
    assert written == [tmp_path / "by_prompt" / "goals_eval" / "n0e1.json"]
