"""simdeploy Stage B2: running a (user_impl.py, user_tests.py) pair. Spec §11 acceptance tests 3
and 13. Runs real pytest subprocesses (sandbox-exec on macOS); no network."""
import shutil
import subprocess
import sys
import time

import pytest

from misalignment_evals.simdeploy.execute import run_tests

if sys.platform == "darwin" and shutil.which("sandbox-exec") is None:
    pytest.skip("sandbox-exec not available", allow_module_level=True)

IMPL = "def add(a, b):\n    return a + b\n"
HEADER = "from user_impl import *\n\n"


def _run(impl: str, tests: str, timeout_s: int = 30):
    return run_tests(impl, HEADER + tests, timeout_s=timeout_s, memory_mb=512)


def test_all_pass():
    r = _run(IMPL, "def test_a():\n    assert add(1, 2) == 3\n")
    assert (r.status, r.test_results) == ("all_pass", {"test_a": "pass"})


def test_some_fail_records_each_test():
    r = _run(IMPL, "def test_a():\n    assert add(1, 2) == 3\n\n"
                   "def test_b():\n    assert add(1, 2) == 4\n")
    assert (r.status, r.test_results) == ("some_fail", {"test_a": "pass", "test_b": "fail"})


def test_collection_error():
    assert _run("def add(a, b)\n    return a + b\n", "def test_a():\n    assert True\n").status \
        == "collection_error"


def test_no_tests_collected():
    assert _run(IMPL, "def helper():\n    return 1\n").status == "no_tests_collected"


def test_exit_at_import_is_not_a_pass():
    """`os._exit(0)` ends the process with code 0 before pytest writes its report."""
    r = _run("import os\nos._exit(0)\n", "def test_a():\n    assert False\n")
    assert r.status == "collection_error"


def test_timeout_terminates_and_leaves_no_process():
    marker = "simdeploy-stray-marker-7f3a"
    tests = (f"import subprocess, sys\n"
             f"def test_hang():\n"
             f"    subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)',\n"
             f"                      '{marker}'])\n"
             f"    while True:\n        pass\n")
    start = time.monotonic()
    r = _run(IMPL, tests, timeout_s=3)
    assert r.status == "timeout"
    assert time.monotonic() - start < 3 + 5
    time.sleep(0.5)
    found = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True).stdout
    assert found.strip() == ""


@pytest.mark.skipif(sys.platform != "darwin", reason="the write/network sandbox is macOS-only")
def test_sandbox_blocks_network_and_outside_writes(tmp_path):
    outside = tmp_path / "escaped.txt"
    tests = ("import socket\n"
             "def test_network():\n"
             "    socket.create_connection(('1.1.1.1', 80), timeout=3)\n\n"
             f"def test_write_outside():\n    open({str(outside)!r}, 'w').write('x')\n\n"
             "def test_write_inside():\n    open('inside.txt', 'w').write('x')\n")
    r = _run(IMPL, tests)
    assert r.test_results == {"test_network": "fail", "test_write_outside": "fail",
                              "test_write_inside": "pass"}
    assert not outside.exists()


def test_same_pair_gives_identical_results():
    tests = ("import pytest\n"
             "def test_a():\n    assert add(1, 2) == 3\n\n"
             "def test_b():\n    assert add(1, 2) == 4\n\n"
             "@pytest.mark.parametrize('x', [1, 2])\ndef test_c(x):\n    assert add(x, 0) == 1\n")
    first, second = _run(IMPL, tests), _run(IMPL, tests)
    assert first.test_results == second.test_results
    assert first.output == second.output
    assert first.test_results == {"test_a": "pass", "test_b": "fail",
                                  "test_c[1]": "pass", "test_c[2]": "fail"}
