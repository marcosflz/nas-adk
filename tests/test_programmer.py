from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from nas.adk.agents import Agent
from nas.adk.models.base_llm import LlmResponse
from nas.adk.runners import Runner
from ejemplos.programmer import agent as programmer
from ejemplos.programmer.tools import MAX_FILE_BYTES, build_tools, resolve_path
from tests.test_runner_contracts import ScriptedLlm, Spy, call


async def test_file_tools_create_read_search_and_safe_edit(tmp_path: Path) -> None:
    tools = {tool.name: tool for tool in build_tools(tmp_path)}
    assert all(not tool.parallel_safe for tool in tools.values())
    await tools["create_file"].run_async(path="src/demo.py", content="value = 1\r\n")
    assert "src/demo.py" in await tools["list_files"].run_async()
    assert "1: value = 1" in await tools["read_file"].run_async(path="src/demo.py")
    assert "src/demo.py:1: value = 1" in await tools["search_text"].run_async(text="value")
    await tools["edit_file"].run_async(path="src/demo.py", old_text="1", new_text="2")
    target = tmp_path / "src/demo.py"
    assert target.read_bytes() == b"value = 2\r\n"
    with pytest.raises(FileExistsError):
        await tools["create_file"].run_async(path="src/demo.py", content="overwrite")
    for old_text in ("", "missing", "a"):
        target.write_bytes(b"aaa\r\n")
        with pytest.raises(ValueError, match="exactamente una vez"):
            await tools["edit_file"].run_async(path="src/demo.py", old_text=old_text, new_text="x")
        assert target.read_bytes() == b"aaa\r\n"
    # También rechaza coincidencias solapadas, no solo las contadas por str.count.
    with pytest.raises(ValueError):
        await tools["edit_file"].run_async(path="src/demo.py", old_text="aa", new_text="x")
    assert not list(target.parent.glob("*.tmp"))


@pytest.mark.parametrize("path", ["../outside.txt", ".git/config", ".env", "keys/private.pem", "a:stream"])
def test_paths_reject_escapes_and_sensitive_files(tmp_path: Path, path: str) -> None:
    with pytest.raises(PermissionError):
        resolve_path(tmp_path, path)
    with pytest.raises(PermissionError):
        resolve_path(tmp_path, str(tmp_path / "absolute.txt"))
    assert resolve_path(tmp_path, "src/../ok.txt") == tmp_path / "ok.txt"
    if os.name == "nt":
        for reserved in ("NUL", "CONOUT$", "COM1.txt"):
            with pytest.raises(PermissionError):
                resolve_path(tmp_path, reserved)


def test_paths_reject_symlinks_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (workspace / "link").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        if os.name != "nt":
            pytest.skip(f"El sistema no permite crear symlinks: {error}")
        # Una junction Windows no necesita el privilegio de crear symlinks.
        result = subprocess.run(
            ["cmd.exe", "/c", "mklink", "/J", str(workspace / "link"), str(outside)],
            capture_output=True, timeout=10,
        )
        assert result.returncode == 0, result.stderr
    with pytest.raises(PermissionError):
        resolve_path(workspace, "link/secret.txt")


async def test_search_exclusions_pagination_and_output_limits(tmp_path: Path) -> None:
    tools = {tool.name: tool for tool in build_tools(tmp_path)}
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git/config").write_text("needle")
    (tmp_path / ".env").write_text("needle")
    (tmp_path / "a.txt").write_text("needle\nsecond\nthird", encoding="utf-8")
    (tmp_path / "b.txt").write_text("needle", encoding="utf-8")
    (tmp_path / "binary.bin").write_bytes(b"needle\x00")
    (tmp_path / "large.txt").write_bytes(b"needle" + b"x" * MAX_FILE_BYTES)
    search = await tools["search_text"].run_async(text="needle")
    assert "a.txt:1:" in search and "b.txt:1:" in search
    assert all(name not in search for name in (".env", ".git", "binary.bin", "large.txt"))
    first_page = await tools["list_files"].run_async(limit=1)
    assert first_page.startswith("a.txt\n") and "offset=1" in first_page
    assert (await tools["list_files"].run_async(offset=1, limit=1)).startswith("b.txt")
    assert "start_line=2" in await tools["read_file"].run_async(path="a.txt", limit=1)
    assert "2: second" in await tools["read_file"].run_async(path="a.txt", start_line=2, limit=1)
    with pytest.raises(ValueError):
        await tools["read_file"].run_async(path="large.txt")
    (tmp_path / "long.txt").write_text("x" * 20000)
    assert "Salida truncada" in await tools["read_file"].run_async(path="long.txt")


async def test_edit_does_not_overwrite_external_change_or_leave_temporary_file(tmp_path: Path) -> None:
    target = tmp_path / "demo.txt"
    target.write_text("before", encoding="utf-8")
    tool = next(tool for tool in build_tools(tmp_path) if tool.name == "edit_file")
    with patch("ejemplos.programmer.tools._read_text", side_effect=["before", "external change"]):
        with pytest.raises(ValueError, match="cambió"):
            await tool.run_async(path="demo.txt", old_text="before", new_text="after")
    assert target.read_text(encoding="utf-8") == "before"
    assert list(tmp_path.iterdir()) == [target]


async def test_dynamic_children_execute_tools_and_return_results(tmp_path: Path) -> None:
    (tmp_path / "demo.txt").write_text("actual file content", encoding="utf-8")
    model = ScriptedLlm(responses=[
        LlmResponse(tool_calls=[call("spawn_programmer", {"task": "Read demo.txt", "name": "child"})]),
        LlmResponse(tool_calls=[call("spawn_programmer", {"task": "Read demo.txt", "name": "grandchild"})]),
        LlmResponse(tool_calls=[call("read_file", {"path": "demo.txt"})]),
        LlmResponse(content="grandchild result"),
        LlmResponse(content="child result"),
        LlmResponse(content="root result"),
    ])
    spy = Spy()
    root = programmer.programmer("programmer", model, build_tools(tmp_path))
    runner = Runner(agent=root, callbacks=[spy])
    events = [event async for event in runner.run_async("user", "session", "PRIVATE_PARENT_HISTORY")]
    assert events[-1].status == "completed" and events[-1].content == "root result"
    assert events[-1].metrics["model_calls"] == 6
    assert model.requests[3].messages[-1]["content"] == "1: actual file content"
    assert model.requests[4].messages[-1]["content"] == "grandchild result"
    assert model.requests[5].messages[-1]["content"] == "child result"
    assert all(
        message.get("content") != "PRIVATE_PARENT_HISTORY"
        for request in model.requests[1:5] for message in request.messages
    )
    assert any(event.author == "grandchild" and event.parent_invocation_id for event in spy.events)
    assert all(not tool.parallel_safe for tool in root.resolve_tools())
    spawn = next(tool for tool in root.resolve_tools() if tool.name == "spawn_programmer")
    assert "ctx" not in spawn.to_llm_tool()["function"]["parameters"]["properties"]


@pytest.mark.parametrize("limits", [{"max_model_calls": 1}, {"max_depth": 0}])
async def test_dynamic_children_share_runner_budgets(tmp_path: Path, limits: dict) -> None:
    model = ScriptedLlm(responses=[
        LlmResponse(tool_calls=[call("spawn_programmer", {"task": "Read", "name": "child"})]),
    ])
    root = programmer.programmer("programmer", model, build_tools(tmp_path))
    runner = Runner(agent=root, **limits)
    events = [event async for event in runner.run_async("user", "session", "Delegate")]
    assert events[-1].status == "budget_exhausted"
    assert len(model.requests) == 1


async def test_commands_require_opt_in_and_run_in_workspace(tmp_path: Path, monkeypatch) -> None:
    tool = next(tool for tool in build_tools(tmp_path) if tool.name == "run_command")
    with patch("ejemplos.programmer.tools.CodeExecutor.run_shell", new_callable=AsyncMock) as shell:
        with pytest.raises(PermissionError, match="ALLOW_COMMANDS"):
            await tool.run_async(command="echo forbidden")
        shell.assert_not_awaited()
    tool = next(tool for tool in build_tools(tmp_path, allow_commands=True) if tool.name == "run_command")
    with pytest.raises(RuntimeError, match="workspace"):
        await tool.run_async(command="echo wrong directory")
    monkeypatch.chdir(tmp_path)
    for timeout in (0, 31, float("nan")):
        with pytest.raises(ValueError):
            await tool.run_async(command="echo invalid", timeout=timeout)
    result = await tool.run_async(
        command=f'"{sys.executable}" -c "from pathlib import Path; print(Path.cwd())"',
    )
    assert result["exit_code"] == 0 and str(tmp_path) in result["stdout"]
    result = await tool.run_async(
        command=f'"{sys.executable}" -c "import time; time.sleep(2)"', timeout=0.1,
    )
    assert result["exit_code"] == -1 and "timed out" in result["stderr"]


@pytest.mark.parametrize("cancelled", [False, True])
async def test_launcher_configures_chat_and_closes_shared_model(
    tmp_path: Path, monkeypatch, capsys, cancelled: bool,
) -> None:
    config = {
        "WORKSPACE": tmp_path, "MODEL": "deepseek-v4.1-flash", "OFFLINE": False,
        "ALLOW_COMMANDS": False, "OPENCODE_API": "responses",
        "GENERATION_CONFIG": {"reasoning_effort": "low"},
        "OPENCODE_BASE_URL": "https://example.invalid/go",
        "OPENCODE_API_KEY": "offline-test-key", "OPENCODE_FALLBACK_MODELS": "backup,other",
        "REQUEST_TIMEOUT": 37.0,
    }
    for name, value in config.items():
        monkeypatch.setattr(programmer.config, name, value)
    monkeypatch.setenv("OPENCODE_API_KEY", "ignored-env-key")
    monkeypatch.chdir(Path.cwd())
    model = ScriptedLlm()

    async def chat(agent: Agent) -> None:
        assert Path.cwd() == tmp_path
        assert agent.model is model
        assert agent.generation_config == config["GENERATION_CONFIG"]
        if cancelled:
            raise asyncio.CancelledError

    with (
        patch.object(programmer, "ModelRouter", return_value=model) as create,
        patch.object(programmer, "run_chat", side_effect=chat) as run_chat,
        patch.object(ScriptedLlm, "aclose", new_callable=AsyncMock) as close,
    ):
        if cancelled:
            with pytest.raises(asyncio.CancelledError):
                await programmer.main()
        else:
            await programmer.main()
        create.assert_called_once_with(
            provider="opencode", base_url=config["OPENCODE_BASE_URL"],
            api_key=config["OPENCODE_API_KEY"], model=config["MODEL"],
            api="responses", fallback_models=("backup", "other"), timeout=37.0,
        )
        run_chat.assert_awaited_once()
        close.assert_awaited_once()
    assert Path.cwd() == tmp_path
    assert "offline-test-key" not in capsys.readouterr().out


async def test_offline_lists_files_without_model_or_writes(tmp_path: Path, monkeypatch, capsys) -> None:
    (tmp_path / "demo.txt").write_text("demo", encoding="utf-8")
    monkeypatch.setattr(programmer.config, "WORKSPACE", tmp_path)
    monkeypatch.setattr(programmer.config, "OFFLINE", True)
    monkeypatch.setattr(programmer.config, "ALLOW_COMMANDS", False)
    monkeypatch.chdir(Path.cwd())
    with (
        patch.object(programmer, "ModelRouter") as create,
        patch.object(programmer, "run_chat") as run_chat,
    ):
        await programmer.main()
        create.assert_not_called()
        run_chat.assert_not_awaited()
    output = capsys.readouterr().out
    assert "OFFLINE" in output and "demo.txt" in output
    assert Path.cwd() == tmp_path
    assert [file.name for file in tmp_path.iterdir()] == ["demo.txt"]
