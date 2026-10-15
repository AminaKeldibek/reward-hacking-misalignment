"""The trusted monitor, run POST-HOC over logged transcripts."""
from pathlib import Path

from control_arena._control_agent import ControlAgent
from control_arena.eval import evaluate_monitors_post_hoc
from control_arena.monitor import (
    MonitorScoreRange,
    basic_monitor_builder,
    full_trajectory_prompt_formatter_markdown,
)
from control_arena.monitor.prompts import monitor_prompts

MONITOR_VIEW = "full_trajectory"


def build_monitor(monitor_model: str) -> ControlAgent:
    """Suspicion monitor over the full trajectory in monitor view: every assistant message's text
    (including <thinking>), tool calls and tool results — not only the latest tool calls. Uses
    control-arena's full-trajectory prompts, which ask for a 0-100 score."""
    return basic_monitor_builder(
        model_role="trusted",
        model=monitor_model,
        system_prompt=monitor_prompts["full_trajectory_default_system"],
        prompt=monitor_prompts["full_trajectory_default_user"],
        prompt_formatter=full_trajectory_prompt_formatter_markdown,
        skip_if_no_tool_calls=False,
        score_range=MonitorScoreRange(min=0.0, max=100.0),
    )


def score_logs(log_dir: "str | Path", monitor_model: str, output_dir: "str | Path") -> list[list[Path]]:
    """Attach suspicion scores to every .eval log under `log_dir`, writing scored copies to
    `output_dir`."""
    log_dir = Path(log_dir)
    out = Path(output_dir).resolve()
    logs = sorted(str(p) for p in log_dir.rglob("*.eval") if out not in p.resolve().parents)
    if not logs:
        raise SystemExit(f"no .eval logs found under {log_dir} — run --mode generate first")
    return evaluate_monitors_post_hoc(
        log_paths=logs, monitors=[build_monitor(monitor_model)], output_dir=str(output_dir)
    )
