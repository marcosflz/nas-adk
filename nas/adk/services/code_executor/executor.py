from __future__ import annotations

import asyncio
import os
import signal
import sys
import tempfile
from pathlib import Path
from typing import cast


class CodeResult:
    """Resultado de código confiable ejecutado en un proceso, no un sandbox."""

    def __init__(self, stdout: str, stderr: str, exit_code: int) -> None:
        self.stdout, self.stderr, self.exit_code = stdout, stderr, exit_code

    @property
    def success(self) -> bool:
        return self.exit_code == 0


class CodeExecutor:
    """Solo código confiable. Un subprocess conserva permisos y entorno del usuario."""

    def __init__(
        self, timeout: float = 30, *, trusted: bool = False, max_output_bytes: int = 65536
    ) -> None:
        if timeout <= 0 or max_output_bytes <= 0:
            raise ValueError("Execution limits must be positive")
        self._timeout, self._trusted, self._max_output = timeout, trusted, max_output_bytes

    def _check_trust(self) -> None:
        if not self._trusted:
            raise PermissionError(
                "CodeExecutor is not a sandbox; opt in with trusted=True for trusted code only"
            )

    async def _kill(self, proc: asyncio.subprocess.Process) -> None:
        if os.name == "nt":
            # ponytail: taskkill alcanza el árbol vivo; usar un Job Object para aislar procesos que se desacoplan.
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(proc.pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
        else:
            try:
                getattr(os, "killpg")(proc.pid, getattr(signal, "SIGKILL"))
            except ProcessLookupError:
                pass
        if proc.returncode is None:
            proc.kill()
        await proc.wait()

    async def _collect(self, proc: asyncio.subprocess.Process) -> CodeResult:
        async def read(stream: asyncio.StreamReader | None) -> str:
            chunks = bytearray()
            truncated = False
            if stream is not None:
                while chunk := await stream.read(8192):
                    remaining = max(0, self._max_output - len(chunks))
                    chunks.extend(chunk[:remaining])
                    truncated |= len(chunk) > remaining
            return chunks.decode(errors="replace") + ("\n[Output truncated]" if truncated else "")

        readers = [asyncio.create_task(read(proc.stdout)), asyncio.create_task(read(proc.stderr))]
        try:
            stdout, stderr, _ = await asyncio.wait_for(
                asyncio.gather(*readers, proc.wait()),
                self._timeout,
            )
            return CodeResult(
                cast(str, stdout),
                cast(str, stderr),
                proc.returncode if proc.returncode is not None else -1,
            )
        except asyncio.TimeoutError:
            await self._kill(proc)
            return CodeResult("", "Execution timed out", -1)
        except asyncio.CancelledError:
            await self._kill(proc)
            raise
        finally:
            for reader in readers:
                if not reader.done():
                    reader.cancel()
            await asyncio.gather(*readers, return_exceptions=True)

    async def run_python(self, code: str) -> CodeResult:
        self._check_trust()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "script.py"
            filepath.write_text(code, encoding="utf-8")
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(filepath),
                cwd=tmpdir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name != "nt",
            )
            return await self._collect(proc)

    async def run_shell(self, command: str) -> CodeResult:
        self._check_trust()
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name != "nt",
        )
        return await self._collect(proc)
