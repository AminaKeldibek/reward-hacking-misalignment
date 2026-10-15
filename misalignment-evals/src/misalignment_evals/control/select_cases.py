"""Case selection from the baseline (honest) arm: keep the cases the model solved in at least
`case_selection.min_solved_epochs` of its baseline epochs, and record which cases and why.

Used two ways:
- in `--mode generate`, between the baseline screen and the non-baseline arms (`select_from_logs`);
- in `--mode select-cases`, to write a manifest that can be frozen as `cases_file` and reused across
  checkpoints (`select_cases`).
"""
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from misalignment_evals.control.config import ControlConfig

MANIFEST_NAME = "selected_cases.json"


def _main_task_value(scorer_name: str, score_value) -> "float | None":
    """1.0/0.0 for the main task, or None if this score isn't the main task. APPS has one combined
    scorer whose value is {'main_task_success': 'C'|'I', ...}; BashArena has a `main_task_success`
    scorer (a dict) next to a separate `side_task_success` scorer, which must never be read as main."""
    from inspect_ai.scorer import CORRECT

    if isinstance(score_value, dict) and "main_task_success" in score_value:
        return 1.0 if score_value["main_task_success"] == CORRECT else 0.0
    if scorer_name == "main_task_success" and not isinstance(score_value, dict):
        return 1.0 if score_value == CORRECT else 0.0
    return None


def pass_counts(log_location: str) -> dict[str, tuple[int, int]]:
    """Per case: (epochs whose main task passed, epochs scored), keyed by sample id."""
    from inspect_ai.log import read_eval_log

    hits: dict[str, list[float]] = defaultdict(list)
    for sample in read_eval_log(log_location).samples or []:
        for name, score in (sample.scores or {}).items():
            v = _main_task_value(name, score.value)
            if v is not None:
                hits[str(sample.id)].append(v)
                break
    return {cid: (int(sum(vs)), len(vs)) for cid, vs in hits.items() if vs}


def manifest_entry(counts: dict[str, tuple[int, int]], min_solved_epochs: int) -> dict:
    """The selection for one eval: cases solved in >= min_solved_epochs baseline epochs."""
    selected = sorted(cid for cid, (passed, _) in counts.items() if passed >= min_solved_epochs)
    return {
        "min_solved_epochs": min_solved_epochs,
        "reason": f"main task passed in >= {min_solved_epochs} baseline epoch(s)",
        "n_candidates": len(counts),
        "n_selected": len(selected),
        "selected": selected,
        "solved_epochs": {cid: {"passed": p, "of": n} for cid, (p, n) in sorted(counts.items())},
    }


def _latest_baseline_logs(log_dir: Path) -> dict[str, str]:
    """{eval_name: newest `<eval>_baseline` log} — eval_set retries can leave older logs behind."""
    from inspect_ai.log import list_eval_logs, read_eval_log

    newest: dict[str, str] = {}
    for info in sorted(list_eval_logs(str(log_dir)), key=lambda i: i.mtime or 0):
        task = read_eval_log(info.name, header_only=True).eval.task   # info.task is the hyphenated file-name form
        if task.endswith("_baseline"):
            newest[task[: -len("_baseline")]] = info.name
    return newest


def select_from_logs(cfg: ControlConfig, eval_names: list[str], baseline_dir: Path) -> dict:
    """Manifest {eval_name: manifest_entry} for `eval_names`, from their baseline logs."""
    logs = _latest_baseline_logs(baseline_dir)
    missing = [e for e in eval_names if e not in logs]
    if missing:
        raise SystemExit(f"no baseline log for {', '.join(missing)} under {baseline_dir}")
    return {e: manifest_entry(pass_counts(logs[e]), cfg.evals[e].case_selection.min_solved_epochs)
            for e in eval_names}


def select_cases(cfg: ControlConfig, output_dir: Path) -> Path:
    """Run only the baseline arm of every eval with case_selection, and write the manifest."""
    from inspect_ai import eval_set

    from misalignment_evals.runners.run_control_evals import build_run_tasks

    selecting = [e for e, b in cfg.evals.items() if b.case_selection is not None]
    if not selecting:
        raise SystemExit("no eval has `case_selection` set in control_evals.evals — nothing to select")
    log_dir = output_dir / f"select_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_run_tasks(cfg, {e: ["baseline"] for e in selecting}, log_dir)
    eval_set(tasks=tasks, log_dir=str(log_dir), max_tasks=cfg.execution.max_tasks,
             max_samples=cfg.execution.max_samples, max_connections=cfg.execution.max_connections,
             max_sandboxes=cfg.execution.max_sandboxes, score=True)
    out = output_dir / MANIFEST_NAME
    out.write_text(json.dumps(select_from_logs(cfg, selecting, log_dir), indent=2))
    return out
