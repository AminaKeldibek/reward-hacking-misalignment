#!/usr/bin/env python3
"""Run reward-hacking evals — measuring test-exploitation ("cheating") in coding agents.

Currently wraps ImpossibleBench.

ImpossibleBench is deliberately NOT in the pinned RL/training deps (it pulls inspect_evals[swe_bench] +
swebench + litellm — heavy, git). Install it into your EVAL environment separately:

    uv pip install "git+https://github.com/safety-research/impossiblebench"
    # the SWE-bench variant additionally needs Docker + the SWE-bench images.

Example (LiveCodeBench, minimal scaffold, no Docker — the recommended MVP):

    uv run --no-sync python reward_hack_evals/run_reward_hack_evals.py \
        --eval impossible_lcb --agent-type minimal \
        --model openai/<served-checkpoint> --model-base-url http://localhost:8001/v1 \
        --api-key inspectai --num-samples 20
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from inspect_ai import eval_set
from misalignment_evals.run_config import load_generation

from rh_model_organism.evals.reward_hack_config import eval_settings, load_reward_hack_config
from rh_model_organism.evals.reward_hack_prompts import (
    HACK_HINTS, TEST_INSTRUCTIONS, apply_impossible_lcb_variant, hint_block, variant_label,
)
from rh_model_organism.evals.secrets import load_secrets_into_env


# Where the model's generated code runs under --sandbox docker. Passed EXPLICITLY: with a bare
# "docker" and no config, inspect searches the WORKING DIRECTORY for a Dockerfile and finds this
# repo's training image — the wrong image, and one that does not build on arm64.
SANDBOX_COMPOSE = Path(__file__).resolve().parent.parent / "reward_hack_evals" / "sandbox" / "compose.yaml"


def resolve_log_dir(output_dir: Path, resume: "str | None") -> Path:
    """Where this run's .eval logs go: a new `logs_<ts>/`, or an existing one to resume into.
    """
    if resume is None:
        return output_dir / f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path(resume)
    if not log_dir.is_dir():
        raise SystemExit(f"--resume {log_dir} is not a directory (expected a logs_<ts> dir)")
    return log_dir


def _require_docker(why: str) -> None:
    """Fail NOW, with a fix, if the Docker daemon is not reachable."""
    if shutil.which("docker") is None:
        raise SystemExit(f"{why}\nDocker is not installed on this machine (no `docker` on PATH).")
    try:
        proc = subprocess.run(["docker", "info"], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SystemExit(f"{why}\nCould not talk to the Docker daemon: {e}") from e
    if proc.returncode != 0:
        detail = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        raise SystemExit(f"{why}\nThe Docker daemon is not running: {detail[-1] if detail else 'docker info failed'}")


def _served_model_name(model: str) -> str:
    """inspect model id -> the name vLLM serves: openai-api/vllm/ckpt400 -> ckpt400, openai/x -> x."""
    parts = model.split("/")
    return "/".join(parts[2:] if parts[0] == "openai-api" else parts[1:])


def _served_api_key(model: str, api_key: "str | None") -> "str | None":
    """The key inspect sends to the server: --api-key, else the provider's env var
    (openai-api/vllm/ckpt400 -> VLLM_API_KEY, openai/x -> OPENAI_API_KEY)."""
    if api_key:
        return api_key
    parts = model.split("/")
    service = parts[1] if parts[0] == "openai-api" else parts[0]
    return os.environ.get(f"{service.upper().replace('-', '_')}_API_KEY")


def _require_tool_calling(model: str, base_url: str, api_key: "str | None") -> None:
    """Fail NOW if the server can't do tool calls. EvilGenie is an agent eval (bash/python/editor/
    submit tools, tool_choice=auto). A vLLM started without --enable-auto-tool-choice 400s every
    request, and EvilGenie's fail_on_error=False then records a "success" with 0 samples scored —
    which is how the 7 Sep olmo_32b runs at ckpt 110/400 came back empty."""
    from openai import AuthenticationError, BadRequestError, OpenAI

    client = OpenAI(base_url=base_url, api_key=_served_api_key(model, api_key) or "EMPTY")
    tool = {"type": "function", "function": {
        "name": "submit", "description": "Submit an answer for evaluation.",
        "parameters": {"type": "object", "properties": {"answer": {"type": "string"}},
                       "required": ["answer"]}}}
    try:
        resp = client.chat.completions.create(
            model=_served_model_name(model), tools=[tool], tool_choice="auto", max_tokens=512,
            temperature=0,
            # A system message on purpose: EvilGenie sends one, and Olmo 3's chat template then
            # lists the tools WITHOUT its "output calls in <function_calls> tags" instruction —
            # the harder case, so test that one.
            messages=[{"role": "system", "content": "You are a skilled software engineer."},
                      {"role": "user", "content": "Call the submit tool with answer '42'."}],
        )
    except BadRequestError as e:
        raise SystemExit(
            f"The model server rejected a tool-calling request:\n  {e}\n"
            "Restart vLLM with tool calling on: set serve.tool_call_parser in the eval config "
            "(olmo3 for Olmo 3, hermes for Qwen3) and re-run serve_eval_checkpoints.sh."
        ) from e
    except AuthenticationError as e:
        raise SystemExit(
            f"The model server rejected the API key:\n  {e}\n"
            "Pass --api-key <serve.api_key>, or set it as VLLM_API_KEY in secrets.json."
        ) from e
    msg = resp.choices[0].message
    if not msg.tool_calls:
        # Server accepts tools but the parser found no call: wrong parser for this model, or the
        # model answers in prose. Either way the agent loop would burn max_messages doing nothing.
        raise SystemExit(
            "Tool-calling preflight: the server accepted the request but returned NO tool call.\n"
            f"  raw content: {(msg.content or '')[:500]!r}\n"
            "Check serve.tool_call_parser matches this model's chat template."
        )
    print(f"Tool-calling preflight OK: {msg.tool_calls[0].function.name}"
          f"({msg.tool_calls[0].function.arguments})")


def _sample_health(log) -> dict:
    """How many samples ran vs errored. A log's status is "success" even when every sample errored
    (fail_on_error=False), so the status alone can't be trusted."""
    from inspect_ai.log import read_eval_log

    if log is None or not getattr(log, "location", None):
        return {"total_samples": 0, "errored_samples": 0, "first_error": None}
    full = read_eval_log(log.location)
    samples = full.samples or []
    errored = [s for s in samples if s.error]
    # Tail, not head: provider errors embed the whole request JSON first and the cause last.
    first = errored[0].error.message.strip()[-300:] if errored else None
    return {"total_samples": len(samples), "errored_samples": len(errored), "first_error": first}


def _load_impossiblebench():
    """Import ImpossibleBench's @tasks, with a clear install hint if it's missing."""
    try:
        from impossiblebench import impossible_livecodebench, impossible_swebench
    except ImportError as e:
        raise SystemExit(
            "ImpossibleBench is not installed. Install into your eval env:\n"
            "    uv pip install 'git+https://github.com/safety-research/impossiblebench'\n"
            f"(import error: {e})"
        )
    return impossible_livecodebench, impossible_swebench


def summarize_cheating(log) -> dict:
    """Pull the cheating rate (= pass rate on impossible tasks) + every scorer metric from an EvalLog."""
    out: dict = {"status": getattr(log, "status", "unknown"), "cheating_rate": None, "metrics": {}}
    results = getattr(log, "results", None) if log else None
    if not results or not getattr(results, "scores", None):
        return out
    score = results.scores[0]
    for name, metric in (score.metrics or {}).items():
        out["metrics"][name] = getattr(metric, "value", None)
    for key in ("accuracy", "mean"):
        if key in out["metrics"]:
            out["cheating_rate"] = out["metrics"][key]
            break
    out["scored_samples"] = getattr(score, "scored_samples", None)
    return out


def summarize_evilgenie(log) -> dict:
    """EvilGenie's scorer emits CATEGORICAL COUNTS (passed_all, llm_reward_hacking, files_modified, …),
    NOT rates — so we divide by scored_samples. Headline reward-hacking signals: llm_reward_hacking
    (judge), files_modified (test-file tamper), passed_visible_only (overfit-to-visible heuristic)."""
    out: dict = {"status": getattr(log, "status", "unknown"), "counts": {}, "rates": {}}
    results = getattr(log, "results", None) if log else None
    if not results or not getattr(results, "scores", None):
        return out
    score = results.scores[0]
    n = getattr(score, "scored_samples", 0) or 0
    for name, metric in (score.metrics or {}).items():
        out["counts"][name] = getattr(metric, "value", None)
    out["scored_samples"] = n
    if n:
        c = out["counts"]
        out["rates"] = {
            "reward_hacking_rate": (c.get("llm_reward_hacking") or 0) / n,
            "files_modified_rate": (c.get("files_modified") or 0) / n,
            "passed_visible_only_rate": (c.get("passed_visible_only") or 0) / n,
            "passed_all_rate": (c.get("passed_all") or 0) / n,
        }
    return out


def build_evilgenie(args):
    """Construct the vendored EvilGenie reward_hacking() task (reward_hack_evals/evilgenie/, MIT)."""
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    eg_dir = str(repo_root / "reward_hack_evals" / "evilgenie")
    if eg_dir not in sys.path:
        sys.path.insert(0, eg_dir)
    try:
        from reward_hacking import reward_hacking
    except ImportError as e:
        raise SystemExit(
            f"EvilGenie failed to import ({e}). Its deps (into your eval env): "
            "inspect_ai, datasets<4, jinja2, openai>=2.14.0, anthropic."
        )
    return reward_hacking(
        difficulty=args.difficulty,
        dataset_source=args.dataset_source,
        limit=args.num_samples,
        enable_llm_judge=not args.no_llm_judge,
        seed=args.seed,
        solver=evilgenie_hint_solver() if args.hack_hints == "on" else None,
    )


def evilgenie_hint_solver():
    """EvilGenie's own default_solver (same tools, same message cap) with the hack hints appended to
    its system message. Rebuilt rather than patched: default_solver hard-codes the message."""
    from constants import DEFAULT_MAX_MESSAGES
    from inspect_ai.solver import basic_agent, system_message
    from inspect_ai.tool import text_editor
    from reward_hacking import DEFAULT_SYSTEM_MESSAGE
    from tools import safe_bash, safe_python

    return basic_agent(
        init=system_message(DEFAULT_SYSTEM_MESSAGE + "\n" + hint_block()),
        tools=[safe_bash(timeout=180), safe_python(), text_editor()],
        max_messages=DEFAULT_MAX_MESSAGES,
    )


# Config keys that map 1:1 onto a CLI flag of the same name. An explicit flag always wins.
_CONFIG_TO_ARG = {
    "samples": "num_samples",
    "epochs": "epochs",
    "agent_type": "agent_type",
    "split": "split",
    "difficulty": "difficulty",
    "dataset_source": "dataset_source",
    "seed": "seed",
    "no_llm_judge": "no_llm_judge",
    "judge_model": "judge_model",
    "sandbox": "sandbox",
    "hack_hints": "hack_hints",
    "test_instruction": "test_instruction",
}

_FALLBACKS = {
    "agent_type": "minimal", "split": "conflicting", "max_connections": 20, "sandbox": "docker",
    "hack_hints": "off", "test_instruction": "dont_modify",
}


def apply_config(args) -> None:
    """Fill unset args from the config's `reward_hacking:` group, then from the built-in fallbacks."""
    if args.config:
        cfg = load_reward_hack_config(args.config)
        settings = eval_settings(cfg, args.eval)
        if args.max_connections is None:
            args.max_connections = cfg["max_connections"]
        for key, dest in _CONFIG_TO_ARG.items():
            if key in settings and getattr(args, dest, None) in (None, False):
                setattr(args, dest, settings[key])
    for dest, fallback in _FALLBACKS.items():
        if getattr(args, dest, None) is None:
            setattr(args, dest, fallback)
    # YAML spells the toggle as a bool; the CLI as on/off.
    if isinstance(args.hack_hints, bool):
        args.hack_hints = "on" if args.hack_hints else "off"
    if args.test_instruction != "dont_modify" and args.eval != "impossible_lcb":
        raise SystemExit(f"test_instruction applies to impossible_lcb only, not {args.eval}")


def build_task(args):
    """Construct the selected reward-hack task (ImpossibleBench or the vendored EvilGenie), with the
    prompt variant applied and named into the task, so two arms of one checkpoint never collide."""
    from inspect_ai import task_with

    label = variant_label(args.eval, args.hack_hints, args.test_instruction)
    if args.eval == "evilgenie":
        task = build_evilgenie(args)
        return task_with(task, name=f"{task.name}_{label}")
    impossible_livecodebench, impossible_swebench = _load_impossiblebench()
    kwargs = {"split": args.split, "agent_type": args.agent_type}
    if args.num_samples is not None:
        kwargs["limit"] = args.num_samples
    if args.eval == "impossible_lcb":
        if args.agent_type != "minimal" and (args.hack_hints == "on" or args.test_instruction == "none"):
            raise SystemExit("hack_hints / test_instruction variants are built for agent_type: minimal only")
        sandbox = ("docker", str(SANDBOX_COMPOSE)) if args.sandbox == "docker" else args.sandbox
        task = impossible_livecodebench(sandbox=sandbox, **kwargs)
        apply_impossible_lcb_variant(task.dataset, args.test_instruction, args.hack_hints)
        return task_with(task, name=f"{task.name}_{label}")
    if args.hack_hints == "on":
        raise SystemExit("hack_hints is not implemented for impossible_swe")
    return impossible_swebench(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Run reward-hacking (test-exploitation) evals")
    parser.add_argument(
        "--eval", choices=["impossible_lcb", "impossible_swe", "evilgenie"], default="impossible_lcb",
        help="Which reward-hack eval to run. impossible_* = ImpossibleBench (pip); evilgenie = "
        "the vendored EvilGenie (needs Docker). Default: impossible_lcb.",
    )
    parser.add_argument("--model", required=True, help="Model to evaluate (e.g. openai/<served>).")
    parser.add_argument("--model-base-url", default=None, help="Base URL for a vLLM server.")
    parser.add_argument("--api-key", default=None, help="API key for the model server (vLLM).")
    parser.add_argument(
        "--config", default=None,
        help="Combined eval config YAML; this eval's budget comes from its `reward_hacking: evals:` "
        "entry (samples/epochs/agent_type/...). CLI flags below override it.",
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Cap #tasks (inspect `limit`).")
    parser.add_argument(
        "--agent-type", default=None,
        help="Scaffold: 'minimal' (LiveCodeBench single-file, no Docker) or 'tools'/'full' (SWE-bench).",
    )
    parser.add_argument(
        "--split", default=None,
        help="[ImpossibleBench] dataset split / impossible variant (default: conflicting).",
    )
    parser.add_argument(
        "--sandbox", default=None, choices=["docker", "local"],
        help="[impossible_lcb] where the model's generated code runs. 'docker' (default, upstream) "
        "isolates it; 'local' runs it in a temp dir ON THIS MACHINE with no isolation — only for a "
        "disposable pod with no Docker daemon. impossible_swe/evilgenie always need Docker.",
    )
    parser.add_argument(
        "--hack-hints", default=None, choices=HACK_HINTS,
        help="'on' adds a system message naming the three RL-trained hacks (always-equal, exit, pytest "
        "report patching) as not allowed — names only, no explanation. Default: off.",
    )
    parser.add_argument(
        "--test-instruction", default=None, choices=TEST_INSTRUCTIONS,
        help="[impossible_lcb] 'none' strips ImpossibleBench's hard-coded '**DO NOT MODIFY THE "
        "TESTS.**' from the prompt and the per-attempt reminder. Default: dont_modify (upstream).",
    )
    # EvilGenie-specific (ignored by impossible_*):
    parser.add_argument(
        "--difficulty", default="hard", choices=["easy", "medium", "hard"],
        help="[EvilGenie] problem difficulty (default: hard).",
    )
    parser.add_argument(
        "--dataset-source", default="livecodebench", choices=["livecodebench", "apps"],
        help="[EvilGenie] dataset (default: livecodebench).",
    )
    parser.add_argument(
        "--no-llm-judge", action="store_true",
        help="[EvilGenie] disable the LLM judge (keep only test-result + file-tamper signals).",
    )
    parser.add_argument(
        "--judge-model", default=None,
        help="[EvilGenie] model for the 'judge' role — overrides its hard-coded openai/gpt-5.",
    )
    parser.add_argument("--seed", type=int, default=42, help="[EvilGenie] dataset split seed (42).")
    parser.add_argument("--epochs", type=int, default=None, help="K attempts per task (inspect epochs).")
    parser.add_argument("--max-connections", type=int, default=None, help="Concurrent connections.")
    parser.add_argument("--output-dir", default="./results/reward_hack", help="Output directory.")
    parser.add_argument(
        "--resume", default=None,
        help="Continue an interrupted run: pass its logs_<ts> dir. Completed samples are skipped "
        "and only the unfinished ones re-run. Default: start a fresh logs_<ts>.",
    )
    parser.add_argument(
        "--retry-attempts", type=int, default=None,
        help="Attempts before eval_set gives up on a failing task (default: inspect's 10).",
    )
    args = parser.parse_args()

    # Judges run HERE, not on the pod: setup.sh's ~/.bashrc loader is the pod's path and
    # zsh never reads it. Without this a key in secrets.json never reaches the runner, and a
    # missing judge key shows up only as an empty logs_<ts>/ (see evilgenie, 4 Sep).
    load_secrets_into_env()
    apply_config(args)

    # Preflight the sandbox BEFORE building anything. Docker is unavoidable for impossible_swe (per-
    # instance images) and evilgenie (its task hardcodes sandbox=("docker", Dockerfile)); for
    # impossible_lcb it is the default but --sandbox local is a way out.
    if args.eval == "impossible_lcb" and args.sandbox == "docker":
        _require_docker(
            "impossible_lcb runs the model's code in a Docker sandbox (ImpossibleBench's default).\n"
            "Either start Docker, or re-run with --sandbox local (no isolation — pod only)."
        )
    elif args.eval in ("impossible_swe", "evilgenie"):
        _require_docker(f"{args.eval} requires Docker (there is no local-sandbox variant).")

    # EvilGenie resolves its judge model role at eval STARTUP. An unresolvable role raises before
    # any sample runs, leaving a logs_<ts>/ with only .eval-set-id — indistinguishable at a glance
    # from a sandbox failure. Fail here instead, naming the fix.
    if args.eval == "evilgenie" and not args.no_llm_judge:
        from inspect_ai.model import get_model

        judge = args.judge_model or "openai/gpt-5"
        try:
            get_model(judge)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                f"{args.eval} needs an LLM judge and {judge!r} will not initialise:\n  {e}\n"
                f"Fix one of: put the key in secrets.json or the environment; pass --judge-model; "
                f"or run with --no-llm-judge to keep only the deterministic signals "
                f"(files_modified + test results)."
            ) from e

    if args.eval == "evilgenie" and args.model_base_url:
        _require_tool_calling(args.model, args.model_base_url, args.api_key)

    task = build_task(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = resolve_log_dir(output_dir, args.resume)
    timestamp = log_dir.name.replace("logs_", "")

    detail = (
        f"difficulty={args.difficulty}, dataset={args.dataset_source}"
        if args.eval == "evilgenie"
        else f"agent_type={args.agent_type}, split={args.split}"
    ) + f", variant={variant_label(args.eval, args.hack_hints, args.test_instruction)}"
    print(f"\n{'=' * 70}")
    print(f"Reward-hacking eval: {args.eval}  ({detail})")
    print(f"Model: {args.model} | samples: {args.num_samples or 'all'} | logs: {log_dir}"
          + ("  (RESUMING)" if args.resume else ""))
    print(f"{'=' * 70}\n")

    eval_kwargs = {"model": args.model}
    if args.model_base_url:
        eval_kwargs["model_base_url"] = args.model_base_url
        if args.api_key:
            # responses_api=False is load-bearing for vLLM-served models: inspect treats any
            # UNKNOWN model name as a new frontier model and routes to /v1/responses, which
            # sends the system prompt as role "developer". Chat templates without a
            # `developer` branch (Olmo's has none) DROP it silently — the model then never
            # sees the reasoning instruction or the scenario framing.
            eval_kwargs["model_args"] = {"api_key": args.api_key, "responses_api": False}
    # EvilGenie's LLM judge uses the "judge" model role (defaults to openai/gpt-5) — override it.
    if args.eval == "evilgenie" and args.judge_model:
        eval_kwargs["model_roles"] = {"judge": args.judge_model}

    optional = {}
    if args.epochs is not None:
        optional["epochs"] = args.epochs
    # ImpossibleBench: fail the run if >10% of samples error. EvilGenie's task sets fail_on_error=False
    # itself (agentic runs error often) — don't override that.
    if args.eval != "evilgenie":
        optional["fail_on_error"] = 0.1

    if args.retry_attempts is not None:
        optional["retry_attempts"] = args.retry_attempts
    # Applied to the model under test only — inspect does not pass it to the EvilGenie judge role.
    generation = load_generation(args.config).model_dump()
    try:
        _success, logs = eval_set(
            tasks=task, log_dir=str(log_dir), max_connections=args.max_connections,
            **generation, **optional, **eval_kwargs,
        )
    finally:
        # inspect creates log_dir before it writes anything into it, so a crash during task/sandbox
        # startup leaves an EMPTY logs_<ts>/ and no summary.json. Say so instead of exiting quietly
        # with a directory that looks like a finished-but-empty run.
        if not any(log_dir.glob("*.eval")):
            print(
                f"\nWARNING: no .eval log was written to {log_dir} — the run failed before any "
                f"sample completed (sandbox/dataset startup). The traceback above is the cause.",
                file=sys.stderr,
            )
    log0 = logs[0] if logs else None

    if args.eval == "evilgenie":
        summary = summarize_evilgenie(log0)
        print(f"\nEvilGenie rates: {summary.get('rates')}")
        print(f"Categorical counts: {summary.get('counts')}")
    else:
        summary = summarize_cheating(log0)
        print(f"\nCHEATING RATE (pass rate on impossible tasks): {summary['cheating_rate']}")
        print(f"All scorer metrics: {summary['metrics']}")

    summary.update(_sample_health(log0))
    if not summary.get("scored_samples"):
        summary["status"] = "failed"
        print(f"\nFAILED: 0 of {summary['total_samples']} samples scored "
              f"({summary['errored_samples']} errored). First error: {summary['first_error']}",
              file=sys.stderr)
    elif summary["errored_samples"]:
        print(f"\nNOTE: {summary['errored_samples']} of {summary['total_samples']} samples errored "
              f"and are excluded from the rates. First error: {summary['first_error']}")

    results = {
        "eval": args.eval, "model": args.model, "num_samples": args.num_samples,
        "epochs": args.epochs, "generation": generation, "timestamp": timestamp, "log_dir": str(log_dir),
        "hack_hints": args.hack_hints,
        "test_instruction": args.test_instruction if args.eval == "impossible_lcb" else None,
        **summary,
    }
    label = variant_label(args.eval, args.hack_hints, args.test_instruction)
    out_file = output_dir / f"reward_hack_{args.eval}_{label}_{args.model.replace('/', '_')}_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    # Also drop the scores INSIDE log_dir so it's a self-contained bundle (the .eval logs carry the
    # per-sample completions + scores; summary.json is the headline). Upload log_dir to keep both.
    with open(log_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")
    print(f"Scores + completions on disk: {log_dir}/  (*.eval logs + summary.json)")
    if results["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
