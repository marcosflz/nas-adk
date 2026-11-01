from __future__ import annotations

import pytest

from nas.adk.services.code_executor.executor import CodeExecutor


@pytest.mark.asyncio
async def test_run_python_success() -> None:
    executor = CodeExecutor(trusted=True)
    result = await executor.run_python("print('hello')")
    assert result.success
    assert "hello" in result.stdout


@pytest.mark.asyncio
async def test_run_python_error() -> None:
    executor = CodeExecutor(trusted=True)
    result = await executor.run_python("raise ValueError('bad')")
    assert not result.success
    assert "ValueError" in result.stderr or "ValueError" in result.stdout


@pytest.mark.asyncio
async def test_run_shell() -> None:
    executor = CodeExecutor(trusted=True)
    result = await executor.run_shell("echo hello")
    assert result.success
    assert "hello" in result.stdout


@pytest.mark.asyncio
async def test_requires_trust_and_bounds_output() -> None:
    with pytest.raises(PermissionError):
        await CodeExecutor().run_python("print('untrusted')")
    result = await CodeExecutor(trusted=True, max_output_bytes=16).run_python("print('x' * 10000)")
    assert result.success and len(result.stdout) < 100 and "truncated" in result.stdout


@pytest.mark.asyncio
async def test_timeout_and_cancellation_clean_up() -> None:
    import asyncio

    executor = CodeExecutor(timeout=0.1, trusted=True)
    result = await executor.run_python("import time; time.sleep(30)")
    assert not result.success and "timed out" in result.stderr
    task = asyncio.create_task(
        CodeExecutor(timeout=30, trusted=True).run_python("import time; time.sleep(30)")
    )
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
