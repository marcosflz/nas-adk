"""Transporte de procesos inyectable para el adaptador ``claude-code``.

La implementación por defecto usa ``asyncio.create_subprocess_exec`` (sin shell). Los tests
inyectan un transporte falso que cumple los mismos protocolos.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Protocol

from nas.adk.models.errors import LlmProcessError

STREAM_LIMIT = 16 * 1024 * 1024
STDERR_TAIL_BYTES = 64 * 1024
TERMINATE_GRACE_SECONDS = 5.0


class ProcessHandle(Protocol):
    """Proceso en ejecución: entrada, salida, cancelación y espera."""

    async def write_stdin(self, data: bytes) -> None:
        """Escribe ``data`` y cierra stdin."""

    def stdout_lines(self) -> AsyncIterator[bytes]:
        """Itera las líneas de stdout."""

    def stderr_tail(self) -> str:
        """Últimos bytes de stderr (acotado)."""

    async def terminate(self) -> None:
        """Termina el proceso (y su grupo) y espera a que salga. Idempotente."""

    async def wait(self) -> int:
        """Espera al proceso y devuelve su código de salida."""


class ProcessTransport(Protocol):
    def which(self, name: str) -> str | None:
        """Localiza un ejecutable o devuelve ``None``."""

    async def spawn(
        self, argv: Sequence[str], *, env: Mapping[str, str], cwd: str
    ) -> ProcessHandle:
        """Lanza el proceso sin shell."""


class _AsyncioHandle:
    def __init__(self, proc: asyncio.subprocess.Process, max_stdout_bytes: int) -> None:
        self._proc = proc
        self._max_stdout = max_stdout_bytes
        self._tail = bytearray()
        self._stderr_task = asyncio.ensure_future(self._drain_stderr())

    async def _drain_stderr(self) -> None:
        stream = self._proc.stderr
        if stream is None:
            return
        try:
            while chunk := await stream.read(8192):
                self._tail += chunk
                if len(self._tail) > STDERR_TAIL_BYTES:
                    del self._tail[: len(self._tail) - STDERR_TAIL_BYTES]
        except (asyncio.CancelledError, ConnectionError):
            pass

    async def write_stdin(self, data: bytes) -> None:
        stdin = self._proc.stdin
        if stdin is None:
            return
        try:
            stdin.write(data)
            await stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass  # el proceso salió antes de leer: el código de salida lo explicará
        finally:
            with contextlib.suppress(Exception):
                stdin.close()

    async def stdout_lines(self) -> AsyncIterator[bytes]:
        stream = self._proc.stdout
        if stream is None:
            return
        total = 0
        while True:
            try:
                line = await stream.readline()
            except (ValueError, asyncio.LimitOverrunError) as error:
                raise LlmProcessError("Claude Code emitted an output line over the size limit") from error
            if not line:
                return
            total += len(line)
            if total > self._max_stdout:
                raise LlmProcessError("Claude Code output exceeded the configured size limit")
            yield line

    def stderr_tail(self) -> str:
        return bytes(self._tail).decode("utf-8", errors="replace")

    def _signal(self, sig: int) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            if os.name == "posix":
                os.killpg(self._proc.pid, sig)
            elif sig == signal.SIGTERM:
                self._proc.terminate()
            else:
                self._proc.kill()

    async def terminate(self) -> None:
        try:
            if self._proc.returncode is None:
                self._signal(signal.SIGTERM)
                try:
                    await asyncio.wait_for(self._proc.wait(), TERMINATE_GRACE_SECONDS)
                except asyncio.TimeoutError:
                    self._signal(getattr(signal, "SIGKILL", signal.SIGTERM))
                    await self._proc.wait()
            elif os.name == "posix":
                self._signal(signal.SIGTERM)  # restos del grupo (nietos huérfanos)
        finally:
            self._stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._stderr_task

    async def wait(self) -> int:
        return await self._proc.wait()


class AsyncioProcessTransport:
    """Transporte real: ``create_subprocess_exec`` sin shell, en su propio grupo de procesos."""

    def __init__(self, max_stdout_bytes: int = 64 * 1024 * 1024) -> None:
        self.max_stdout_bytes = max_stdout_bytes

    def which(self, name: str) -> str | None:
        return shutil.which(name)

    async def spawn(
        self, argv: Sequence[str], *, env: Mapping[str, str], cwd: str
    ) -> ProcessHandle:
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=dict(env),
                cwd=cwd,
                limit=STREAM_LIMIT,
                start_new_session=os.name == "posix",
            )
        except (FileNotFoundError, PermissionError) as error:
            raise LlmProcessError(f"Cannot execute {argv[0]!r}: {error.strerror}") from error
        return _AsyncioHandle(proc, self.max_stdout_bytes)


async def capture(
    transport: ProcessTransport,
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    cwd: str,
    timeout: float,
    stdin: bytes = b"",
) -> tuple[int, str, str]:
    """Ejecuta un comando corto y devuelve ``(código, stdout, stderr_tail)``."""
    handle = await transport.spawn(argv, env=env, cwd=cwd)
    try:
        async def _run() -> tuple[int, str]:
            await handle.write_stdin(stdin)
            chunks = [line async for line in handle.stdout_lines()]
            code = await handle.wait()
            return code, b"".join(chunks).decode("utf-8", errors="replace")

        code, out = await asyncio.wait_for(_run(), timeout)
        return code, out, handle.stderr_tail()
    finally:
        await handle.terminate()
