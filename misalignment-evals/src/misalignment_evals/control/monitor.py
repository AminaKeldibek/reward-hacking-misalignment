"""The trusted monitor, run POST-HOC over logged transcripts."""
from pathlib import Path

from control_arena._control_agent import ControlAgent
from control_arena.eval import evaluate_monitors_post_hoc
from control_arena.monitor import basic_monitor_builder


def build_monitor(monitor_model: str) -> ControlAgent:
    """A basic suspicion monitor backed by the given (trusted) judge model. Default rubric: read the
    agent's tool calls, emit a 0-10 suspicion score (control-arena normalizes to 0-1)."""
    return basic_monitor_builder(model_role="trusted", model=monitor_model)


def score_logs(log_dir: "str | Path", monitor_model: str, output_dir: "str | Path") -> list[list[Path]]:
    """Attach suspicion scores to every .eval log under `log_dir`, writing scored copies to
    `output_dir`."""
    log_dir = Path(log_dir)
    logs = sorted(str(p) for p in log_dir.glob("*.eval"))
    if not logs:
        raise SystemExit(f"no .eval logs found under {log_dir} — run --mode generate first")
    return evaluate_monitors_post_hoc(
        log_paths=logs, monitors=[build_monitor(monitor_model)], output_dir=str(output_dir)
    )
