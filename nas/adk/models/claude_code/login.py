"""Vinculación explícita de una suscripción con ``claude setup-token`` (CLI oficial).

NAS no implementa el OAuth de Anthropic ni toca credenciales guardadas: lanza el comando
oficial en un pseudo-terminal con un ``HOME`` temporal y vacío, devuelve la URL que el propio
CLI muestra, le pasa el código que el titular de la cuenta pega tras autorizar y recoge el token
de larga duración que imprime. Al terminar se borra el ``HOME`` temporal.

Solo se ejecuta cuando la aplicación lo pide de forma explícita (por ejemplo, el usuario pulsa
«Vincular cuenta»). El token resultante pertenece a quien inició sesión: guárdalo asociado a esa
persona y no lo uses para atender peticiones de otras.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import os
import re
import shutil
import signal
import struct
import tempfile
import termios
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Protocol

from nas.adk.models.claude_code import isolation
from nas.adk.models.errors import LlmAuthError, LlmProcessError, LlmTimeoutError

LOGIN_TTL_SECONDS = 600.0
URL_TIMEOUT_SECONDS = 45.0
TOKEN_TIMEOUT_SECONDS = 60.0
_MAX_BUFFER = 256 * 1024

# Movimientos de cursor horizontales (`ESC[nG`, `ESC[nC`) separan palabras en la salida del CLI.
_CURSOR_GAP = re.compile(r"\x1b\[\d*[GC]")
_OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_CSI = re.compile(r"\x1b\[[0-9;?<>=!]*[ -/]*[@-~]")
_ESC = re.compile(r"\x1b[@-Z\\-_78]")
_URL_RE = re.compile(r"https://(?:claude\.com|claude\.ai)/[^\s\"'<>]*oauth/authorize\?[^\s\"'<>]+")
_TOKEN_RE = re.compile(r"sk-ant-oat\d{2}-[A-Za-z0-9_\-]{20,}")
_ERROR_RE = re.compile(r"(?i)\b(error|invalid|failed|expired)\b[^\r\n]*")


def clean_terminal_output(raw: str) -> str:
    """Quita secuencias de escape del terminal y conserva el texto legible."""
    text = _OSC.sub("", raw)
    text = _CURSOR_GAP.sub(" ", text)
    text = _CSI.sub("", text)
    text = _ESC.sub("", text)
    return text.replace("\r", "")


def find_login_url(raw: str) -> str | None:
    match = _URL_RE.search(clean_terminal_output(raw))
    return match.group(0) if match else None


def find_token(raw: str) -> str | None:
    match = _TOKEN_RE.search(clean_terminal_output(raw))
    return match.group(0) if match else None


class PtyProcess(Protocol):
    """Proceso conectado a un pseudo-terminal."""

    async def read(self, timeout: float) -> bytes | None:
        """Bytes disponibles (``b""`` si no llegó nada a tiempo) o ``None`` si el proceso salió."""

    async def write(self, data: bytes) -> None:
        """Escribe en el terminal del proceso."""

    async def terminate(self) -> None:
        """Termina el proceso (y su grupo). Idempotente."""


PtySpawner = Callable[[Sequence[str], Mapping[str, str], str], Awaitable[PtyProcess]]


class _AsyncioPty:
    def __init__(self, proc: asyncio.subprocess.Process, master: int) -> None:
        self._proc = proc
        self._master = master
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._closed = False
        loop = asyncio.get_running_loop()
        os.set_blocking(master, False)
        loop.add_reader(master, self._on_readable)

    def _on_readable(self) -> None:
        try:
            data = os.read(self._master, 65536)
        except BlockingIOError:
            return
        except OSError:
            data = b""
        if not data:
            self._stop_reading()
            self._queue.put_nowait(None)
        else:
            self._queue.put_nowait(data)

    def _stop_reading(self) -> None:
        if not self._closed:
            self._closed = True
            with contextlib.suppress(Exception):
                asyncio.get_running_loop().remove_reader(self._master)

    async def read(self, timeout: float) -> bytes | None:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout)
        except asyncio.TimeoutError:
            return b""

    async def write(self, data: bytes) -> None:
        view = memoryview(data)
        while view:
            try:
                written = os.write(self._master, view)
            except BlockingIOError:
                await asyncio.sleep(0.01)
                continue
            view = view[written:]

    async def terminate(self) -> None:
        self._stop_reading()
        if self._proc.returncode is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self._proc.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(self._proc.wait(), 5)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(self._proc.pid, signal.SIGKILL)
                await self._proc.wait()
        with contextlib.suppress(OSError):
            os.close(self._master)


async def spawn_pty(argv: Sequence[str], env: Mapping[str, str], cwd: str) -> PtyProcess:
    """Lanza ``argv`` sin shell en un pseudo-terminal ancho (las URL no se parten)."""
    if os.name != "posix":
        raise LlmProcessError("Linking a Claude subscription requires a POSIX pseudo-terminal")
    master, slave = os.openpty()
    try:
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 50, 1000, 0, 0))
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env=dict(env),
            cwd=cwd,
            start_new_session=True,
        )
    except (FileNotFoundError, PermissionError) as error:
        os.close(master)
        raise LlmProcessError(f"Cannot execute {argv[0]!r}: {error.strerror}") from error
    finally:
        os.close(slave)
    return _AsyncioPty(proc, master)


def login_env(parent: Mapping[str, str], home: str) -> dict[str, str]:
    """Entorno del login: sin credenciales previas, sin navegador y con ``HOME`` propio."""
    env = isolation.build_child_env(parent, policy="scrub")
    env.pop(isolation.OAUTH_TOKEN_VAR, None)
    for name in ("DISPLAY", "WAYLAND_DISPLAY", "XDG_CONFIG_HOME", "CLAUDE_CONFIG_DIR"):
        env.pop(name, None)
    env.update(HOME=home, BROWSER="true", TERM="xterm-256color", COLUMNS="1000", LINES="50")
    return env


class SetupTokenSession:
    """Una vinculación en curso: ``start()`` → URL, ``submit(code)`` → token, ``close()``.

    Úsala desde una acción explícita del titular de la cuenta. Ciérrala siempre (también se
    puede usar como gestor de contexto asíncrono).
    """

    def __init__(
        self,
        *,
        executable: str | None = None,
        spawn: PtySpawner | None = None,
        which: Callable[[str], str | None] = shutil.which,
        environ: Mapping[str, str] | None = None,
        ttl: float = LOGIN_TTL_SECONDS,
    ) -> None:
        self.executable = executable or (environ or os.environ).get("CLAUDE_CODE_BIN") or "claude"
        self._spawn = spawn or spawn_pty
        self._which = which
        self._environ = os.environ if environ is None else environ
        self.expires_at = time.monotonic() + ttl
        self._home: str | None = None
        self._proc: PtyProcess | None = None
        self._buffer = ""
        self._submitted = False
        self.url: str | None = None

    @property
    def expired(self) -> bool:
        return time.monotonic() >= self.expires_at

    async def __aenter__(self) -> SetupTokenSession:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def _read_until(self, pick: Callable[[str], str | None], timeout: float) -> str | None:
        assert self._proc is not None
        deadline = time.monotonic() + timeout
        while True:
            found = pick(self._buffer)
            if found:
                return found
            left = deadline - time.monotonic()
            if left <= 0:
                raise LlmTimeoutError("Claude Code did not answer in time while linking the account")
            chunk = await self._proc.read(min(left, 1.0))
            if chunk is None:
                return pick(self._buffer)
            if chunk:
                self._buffer = (self._buffer + chunk.decode("utf-8", errors="replace"))[-_MAX_BUFFER:]

    async def start(self) -> str:
        """Lanza ``claude setup-token`` y devuelve la URL de autorización que muestra."""
        if self._proc is not None:
            raise LlmProcessError("This login session was already started")
        path = self._which(self.executable)
        if not path:
            raise LlmProcessError(f"Claude Code executable {self.executable!r} was not found")
        self._home = tempfile.mkdtemp(prefix="nas-cc-login-")
        try:
            env = login_env(self._environ, self._home)
            self._proc = await self._spawn([path, "setup-token"], env, self._home)
            url = await self._read_until(find_login_url, URL_TIMEOUT_SECONDS)
        except BaseException:
            await self.close()
            raise
        if not url:
            tail = isolation.redact(clean_terminal_output(self._buffer))
            await self.close()
            raise LlmProcessError(f"`claude setup-token` did not show a sign-in URL: {tail}")
        self.url = url
        return url

    async def submit(self, code: str) -> str:
        """Pasa al CLI el código de autorización y devuelve el token de larga duración."""
        code = code.strip()
        if self._proc is None or self.url is None:
            raise LlmProcessError("The login session is not started")
        if self._submitted:
            raise LlmProcessError("A code was already submitted for this login session")
        if self.expired:
            await self.close()
            raise LlmTimeoutError("The login session expired; start it again")
        if not code or len(code) > 512 or any(ch.isspace() or ord(ch) < 32 for ch in code):
            raise LlmAuthError("The authorization code is not valid")
        self._submitted = True
        mark = len(self._buffer)
        try:
            # Pegado entre corchetes (bracketed paste) y Enter aparte, como haría un terminal.
            await self._proc.write(b"\x1b[200~" + code.encode() + b"\x1b[201~")
            await asyncio.sleep(0.3)
            await self._proc.write(b"\r")

            def after(buffer: str) -> str | None:
                fresh = buffer[mark:]
                token = find_token(fresh)
                if token:
                    return token
                error = _ERROR_RE.search(clean_terminal_output(fresh))
                return f"!{error.group(0)}" if error else None

            found = await self._read_until(after, TOKEN_TIMEOUT_SECONDS)
        finally:
            await self.close()
        if not found or found.startswith("!"):
            detail = isolation.redact(found[1:] if found else clean_terminal_output(self._buffer[mark:]))
            raise LlmAuthError(f"Anthropic did not accept the authorization code: {detail}")
        return found

    async def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            with contextlib.suppress(Exception):
                await proc.terminate()
        home, self._home = self._home, None
        if home:
            shutil.rmtree(home, ignore_errors=True)
