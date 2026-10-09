"""Run a (user_impl.py, user_tests.py) pair under pytest in a throwaway directory
(spec §5 B2, §8.3).

Isolation is a local subprocess, not a container:
- macOS: wrapped in `sandbox-exec` with a profile that denies network access and any file write
  outside the run's own directory. macOS cannot cap memory, so `memory_mb` is not enforced there.
- Linux: no filesystem/network sandbox; address space is capped at `memory_mb` with RLIMIT_AS.
The whole process group is killed after the run, on timeout or not, so nothing outlives it.
"""
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Literal

IMPL_FILE = "user_impl.py"
TESTS_FILE = "user_tests.py"
REPORT_FILE = "report.xml"
PYTEST_ARGS = ["-q", "--tb=short", "-p", "no:cacheprovider", f"--junitxml={REPORT_FILE}",
               TESTS_FILE]
OUTPUT_TAIL = 4000
_DURATION = re.compile(r" in \d+(?:\.\d+)?s")

RunStatus = Literal["all_pass", "some_fail", "collection_error", "timeout", "no_tests_collected"]
TestOutcome = Literal["pass", "fail", "error", "skipped"]


@dataclass
class RunResult:
    status: RunStatus
    test_results: dict[str, TestOutcome]
    output: str      # pytest's stdout tail, durations and the run directory stripped


def _profile(workdir: str) -> str:
    return ("(version 1)(allow default)(deny network*)(deny file-write*)"
            f'(allow file-write* (subpath "{workdir}") (literal "/dev/null") (literal "/dev/tty"))')


def _command(workdir: str, memory_mb: int) -> list[str]:
    cmd = [sys.executable, "-m", "pytest", *PYTEST_ARGS]
    if sys.platform == "darwin":
        return ["sandbox-exec", "-p", _profile(workdir), *cmd]
    return ["sh", "-c", f'ulimit -v {memory_mb * 1024} && exec "$@"', "sh", *cmd]


def _environment(workdir: str) -> dict[str, str]:
    tmp = os.path.join(workdir, "tmp")
    os.makedirs(tmp, exist_ok=True)
    return {"PATH": "/usr/bin:/bin", "HOME": workdir, "TMPDIR": tmp, "PYTHONHASHSEED": "0",
            "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _test_key(case: ET.Element) -> str:
    """`test_x` for a module-level test, `TestCls::test_x` for a method."""
    classname, name = case.get("classname") or "", case.get("name") or ""
    module_prefix = TESTS_FILE.removesuffix(".py")
    if classname in ("", module_prefix):
        return name
    return f"{classname.removeprefix(module_prefix + '.')}::{name}"


def _outcome(case: ET.Element) -> TestOutcome:
    tags = {child.tag for child in case}
    if "error" in tags:
        return "error"
    if "failure" in tags:
        return "fail"
    return "skipped" if "skipped" in tags else "pass"


def parse_report(path: Path) -> "dict[str, TestOutcome] | None":
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return None
    return {_test_key(case): _outcome(case) for case in root.iter("testcase")}


def _status(returncode: int, results: "dict[str, TestOutcome] | None") -> RunStatus:
    """pytest exits 0 all passed, 1 some failed, 5 nothing collected, 2-4 interrupted/error.
    A run that left no report (the process exited before pytest wrote it, e.g. `os._exit` at
    import) is not a pass, whatever its exit code."""
    if returncode == 5:
        return "no_tests_collected"
    if results is None or returncode not in (0, 1):
        return "collection_error"
    if not results:
        return "no_tests_collected"
    passed = returncode == 0 and all(v == "pass" for v in results.values())
    return "all_pass" if passed else "some_fail"


def _clean(output: str, workdir: str) -> str:
    output = output.replace(workdir, ".").replace(os.path.realpath(workdir), ".")
    return _DURATION.sub("", output)[-OUTPUT_TAIL:]


def _after_kill(proc: subprocess.Popen) -> bytes:
    """Whatever output is left once the group is dead. A process that escaped the group can keep
    the pipe open, so give up after a few seconds rather than hang."""
    try:
        return proc.communicate(timeout=5)[0] or b""
    except subprocess.TimeoutExpired:
        proc.kill()
        return b""


def run_tests(impl: str, tests: str, timeout_s: int, memory_mb: int) -> RunResult:
    workdir = os.path.realpath(tempfile.mkdtemp(prefix="simdeploy_"))
    try:
        Path(workdir, IMPL_FILE).write_text(impl)
        Path(workdir, TESTS_FILE).write_text(tests)
        proc = subprocess.Popen(
            _command(workdir, memory_mb), cwd=workdir, env=_environment(workdir),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True)
        try:
            out, _ = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_group(proc.pid)
            leftover = _after_kill(proc).decode(errors="replace")
            return RunResult("timeout", {}, _clean(leftover, workdir))
        finally:
            _kill_group(proc.pid)
        results = parse_report(Path(workdir, REPORT_FILE))
        return RunResult(_status(proc.returncode, results), results or {},
                         _clean(out.decode(errors="replace"), workdir))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def sandbox_description(timeout_s: int, memory_mb: int, packages: list[str]) -> dict:
    """What the code ran on. Its hash stands in for the spec's `sandbox_image_digest`."""
    versions = {}
    for name in sorted({"pytest", *packages}):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": sys.platform,
            "isolation": "sandbox-exec" if sys.platform == "darwin" else "rlimit",
            "packages": versions, "pytest_args": PYTEST_ARGS,
            "timeout_s": timeout_s, "memory_mb": memory_mb}


def sandbox_digest(description: dict) -> str:
    blob = json.dumps(description, sort_keys=True).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()
