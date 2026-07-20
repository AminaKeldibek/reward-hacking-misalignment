import asyncio
import subprocess
from pathlib import Path

from inspect_ai.util import ExecResult, SandboxEnvironment


class FastLocalSandbox(SandboxEnvironment):
    """Local sandbox for RL scoring."""

    def __init__(self, directory: str):
        self._dir = Path(directory)

    def _resolve(self, p: "str | None") -> Path:
        if p is None or p == ".":
            return self._dir
        pp = Path(p)
        return pp if pp.is_absolute() else self._dir / pp

    async def exec(self, cmd, input=None, cwd=None, env=None, user=None,
                   timeout=None, timeout_retry=True, concurrency=True) -> ExecResult[str]:
        def _run():
            return subprocess.run(
                cmd, cwd=self._resolve(cwd), env=env,
                input=input.encode() if isinstance(input, str) else input,
                capture_output=True, timeout=timeout,
            )
        try:
            cp = await asyncio.to_thread(_run)
        except subprocess.TimeoutExpired as e:
            return ExecResult(success=False, returncode=124,
                              stdout=(e.stdout or b"").decode("utf-8", "replace"),
                              stderr="TIMEOUT")
        return ExecResult(success=cp.returncode == 0, returncode=cp.returncode,
                          stdout=cp.stdout.decode("utf-8", "replace"),
                          stderr=cp.stderr.decode("utf-8", "replace"))

    async def write_file(self, file: str, contents) -> None:
        path = self._resolve(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w" if isinstance(contents, str) else "wb") as f:
            f.write(contents)

    async def read_file(self, file: str, text: bool = True):
        with open(self._resolve(file), "r" if text else "rb") as f:
            return f.read()

    @classmethod
    async def sample_cleanup(cls, task_name, config, environments, interrupted) -> None:
        # Required by the SandboxEnvironment ABC (it's the 4th abstract method, alongside
        # exec/read_file/write_file) but never reached in our flow: scoring._score_one owns each
        # sandbox's temp dir via tempfile.TemporaryDirectory and binds it through the ContextVar
        # directly, so inspect's init/cleanup lifecycle never runs. Stub to satisfy the ABC.
        return None