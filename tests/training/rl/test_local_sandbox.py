"""Unit tests for FastLocalSandbox — the plain temp-dir + subprocess sandbox that replaces
inspect's local sandbox for RL scoring (see md_files/retro.md + wiki "Why we keep inspect...").

These run real subprocesses on the host — no GPU, no Docker, no inspect sandbox context. They're
the "Full-B is locally testable" payoff: the exec path that used to deadlock on the pod is now
plain stdlib we can exercise on a laptop.
"""
import asyncio
import sys
import time

import pytest

pytest.importorskip("inspect_ai")

from inspect_ai.util import ExecResult, SandboxEnvironment  # noqa: E402

from rh_model_organism.training.rl.local_sandbox import FastLocalSandbox  # noqa: E402


def _run(coro):
    """Drive one coroutine to completion (the sandbox API is async; these tests are sync)."""
    return asyncio.run(coro)


def test_is_a_sandbox_environment(tmp_path):
    # It must subclass inspect's SandboxEnvironment so the scorers' sandbox() call accepts it.
    assert isinstance(FastLocalSandbox(str(tmp_path)), SandboxEnvironment)


def test_write_then_exec_returns_execresult(tmp_path):
    sb = FastLocalSandbox(str(tmp_path))
    _run(sb.write_file("hello.py", "print('hi from sandbox')"))
    res = _run(sb.exec([sys.executable, "hello.py"], cwd="."))
    assert isinstance(res, ExecResult)         # the exact type the scorers read .success/.stdout off
    assert res.success is True
    assert res.returncode == 0
    assert "hi from sandbox" in res.stdout


def test_exec_nonzero_returncode_is_not_success(tmp_path):
    sb = FastLocalSandbox(str(tmp_path))
    res = _run(sb.exec([sys.executable, "-c", "import sys; sys.exit(3)"]))
    assert res.success is False
    assert res.returncode == 3


def test_exec_timeout_returns_fast_and_flags_failure(tmp_path):
    # The whole point of the per-exec timeout: a runaway process fails loudly, never hangs.
    sb = FastLocalSandbox(str(tmp_path))
    t0 = time.time()
    res = _run(sb.exec([sys.executable, "-c", "import time; time.sleep(30)"], timeout=1))
    elapsed = time.time() - t0
    assert res.success is False
    assert res.returncode == 124
    assert res.stderr == "TIMEOUT"
    assert elapsed < 10, f"timeout should return promptly, took {elapsed:.1f}s"


def test_cwd_dot_resolves_to_the_sandbox_dir(tmp_path):
    # "." (what the scorers pass as workdir on the local backend) must map to our temp dir.
    sb = FastLocalSandbox(str(tmp_path))
    _run(sb.write_file("marker.txt", "x"))
    res = _run(sb.exec(["ls"], cwd="."))
    assert "marker.txt" in res.stdout


def test_write_file_creates_parent_dirs(tmp_path):
    sb = FastLocalSandbox(str(tmp_path))
    _run(sb.write_file("nested/deep/x.txt", "data"))
    assert (tmp_path / "nested" / "deep" / "x.txt").read_text() == "data"


def test_text_and_binary_roundtrip(tmp_path):
    sb = FastLocalSandbox(str(tmp_path))
    _run(sb.write_file("r.txt", "roundtrip"))
    assert _run(sb.read_file("r.txt")) == "roundtrip"
    _run(sb.write_file("b.bin", b"\x00\x01\x02"))
    assert _run(sb.read_file("b.bin", text=False)) == b"\x00\x01\x02"


def test_exec_honours_input_and_env(tmp_path):
    sb = FastLocalSandbox(str(tmp_path))
    # stdin is forwarded...
    res = _run(sb.exec([sys.executable, "-c", "import sys; print(sys.stdin.read().upper())"],
                       input="hello"))
    assert "HELLO" in res.stdout
    # ...and so is a custom env.
    res = _run(sb.exec([sys.executable, "-c", "import os; print(os.environ['RH_MARKER'])"],
                       env={"RH_MARKER": "present", "PATH": ""}))
    assert "present" in res.stdout
