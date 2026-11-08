from __future__ import annotations

import asyncio
import os
import sys
import time

import pytest

from nas.adk.models import LlmProcessError
from nas.adk.models.claude_code.transport import AsyncioProcessTransport, capture


async def test_large_stdout_and_stderr_do_not_block() -> None:
    code = (
        "import sys\n"
        "sys.stderr.write('e' * 500000)\n"
        "sys.stdout.write('x' * 200000 + '\\n')\n"
        "sys.stdout.write('fin\\n')\n"
    )
    transport = AsyncioProcessTransport()
    status, out, err = await asyncio.wait_for(
        capture(transport, [sys.executable, "-c", code], env=os.environ, cwd="/tmp", timeout=20),
        30,
    )
    assert status == 0 and out.endswith("fin\n") and len(out) > 200000
    assert 0 < len(err) <= 64 * 1024  # stderr acotado


async def test_stdin_is_delivered_without_shell() -> None:
    code = "import sys; print(sys.stdin.read().upper())"
    status, out, _ = await capture(
        AsyncioProcessTransport(),
        [sys.executable, "-c", code],
        env=os.environ,
        cwd="/tmp",
        timeout=20,
        stdin=b"hola; $(echo pwned) `id`",
    )
    assert out.strip() == "HOLA; $(ECHO PWNED) `ID`"  # sin interpolación de shell


async def test_stdout_limit_is_enforced() -> None:
    transport = AsyncioProcessTransport(max_stdout_bytes=1000)
    with pytest.raises(LlmProcessError, match="size limit"):
        await capture(
            transport,
            [sys.executable, "-c", "print('y' * 5000)"],
            env=os.environ,
            cwd="/tmp",
            timeout=20,
        )


async def test_missing_executable_is_typed_error() -> None:
    with pytest.raises(LlmProcessError, match="Cannot execute"):
        await AsyncioProcessTransport().spawn(["/no/existe/claude"], env={}, cwd="/tmp")


@pytest.mark.skipif(os.name != "posix", reason="grupos de procesos POSIX")
async def test_terminate_kills_whole_process_group() -> None:
    # El hijo lanza un nieto que ignora SIGTERM; terminate() debe acabar con ambos.
    code = (
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', "
        "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print(1, flush=True); time.sleep(60)'])\n"
        "print(p.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    transport = AsyncioProcessTransport()
    handle = await transport.spawn([sys.executable, "-c", code], env=os.environ, cwd="/tmp")
    pid = 0
    async for line in handle.stdout_lines():
        pid = int(line.strip())
        break
    import nas.adk.models.claude_code.transport as mod

    original = mod.TERMINATE_GRACE_SECONDS
    mod.TERMINATE_GRACE_SECONDS = 0.5
    try:
        started = time.monotonic()
        await handle.terminate()
        assert time.monotonic() - started < 10
    finally:
        mod.TERMINATE_GRACE_SECONDS = original
    await asyncio.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
