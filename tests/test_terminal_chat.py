from __future__ import annotations

import asyncio
import io
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from click.testing import CliRunner
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown

from nas.adk import Agent, Runner
from nas.adk.cli import run_chat
from nas.adk.cli.main import main as cli
from nas.adk.cli.main import _chat_turn
from nas.adk.models import ModelRouter
from nas.adk.models.base_llm import LlmResponse
from nas.adk.sessions import InMemorySessionService
from tests.test_runner_contracts import ScriptedLlm, call

EXAMPLE = Path(__file__).resolve().parents[1] / "ejemplos/investigador-terminal"


def test_python_launcher_offline_from_another_directory(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(EXAMPLE / "agent.py"), "--offline"],
        input="cambio climático\n/reset\nenergía solar\n/salir\n",
        encoding="utf-8", capture_output=True, cwd=tmp_path, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "Modo offline" in result.stdout and "Datos FICTICIOS" in result.stdout
    assert result.stdout.count("=== INFORME NAS ===") == 2
    assert "Conversación reiniciada" in result.stdout and "Tú >" in result.stdout
    assert "cambio climático" in result.stdout and "3.2TW" in result.stdout


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("api", ["responses", "chat_completions", "auto"])
def test_agent_main_configures_opencode_and_closes_model(monkeypatch, capsys, cancelled, api) -> None:
    entry = runpy.run_path(str(EXAMPLE / "agent.py"))["main"]
    assert not capsys.readouterr().out  # Importar no inicia el chat.
    config = {
        "MODEL": "deepseek-v4.1-flash",
        "OPENCODE_API": api,
        "GENERATION_CONFIG": {"reasoning_effort": "none"},
        "OPENCODE_BASE_URL": "https://example.invalid/go",
        "OPENCODE_API_KEY": "offline-test-key",
        "OPENCODE_FALLBACK_MODELS": "backup-model",
        "REQUEST_TIMEOUT": 37.0,
    }
    for name, value in config.items():
        monkeypatch.setitem(entry.__globals__, name, value)
    monkeypatch.setenv("OPENCODE_BASE_URL", "https://ignored.invalid/go")
    monkeypatch.setenv("OPENCODE_API_KEY", "ignored-key")
    monkeypatch.setattr(sys, "argv", [str(EXAMPLE / "agent.py")])
    chat = AsyncMock(side_effect=asyncio.CancelledError() if cancelled else None)
    monkeypatch.setitem(entry.__globals__, "run_chat", chat)
    model = ScriptedLlm()
    create_model = Mock(return_value=model)
    monkeypatch.setitem(entry.__globals__, "ModelRouter", create_model)
    with (
        patch.object(ScriptedLlm, "aclose", new_callable=AsyncMock) as close,
    ):
        if cancelled:
            with pytest.raises(asyncio.CancelledError):
                entry()
        else:
            entry()
        create_model.assert_called_once_with(
            provider="opencode", base_url=config["OPENCODE_BASE_URL"],
            api_key=config["OPENCODE_API_KEY"], model=config["MODEL"], api=api,
            fallback_models=("backup-model",), timeout=config["REQUEST_TIMEOUT"],
        )
        assert entry.__globals__["root_agent"].generation_config == config["GENERATION_CONFIG"]
        chat.assert_awaited_once_with(entry.__globals__["root_agent"])
        close.assert_awaited_once()
    assert "offline-test-key" not in capsys.readouterr().out


def test_agent_main_rejects_missing_config_without_network(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", [str(EXAMPLE / "agent.py")])
    monkeypatch.delenv("OPENCODE_BASE_URL", raising=False)
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    entry = runpy.run_path(str(EXAMPLE / "agent.py"))["main"]
    monkeypatch.setitem(entry.__globals__, "OPENCODE_BASE_URL", "https://example.invalid/go")
    monkeypatch.setitem(entry.__globals__, "OPENCODE_API_KEY", "")
    create_model = Mock()
    monkeypatch.setitem(entry.__globals__, "ModelRouter", create_model)
    with pytest.raises(SystemExit) as error:
        entry()
    assert error.value.code == 2
    create_model.assert_not_called()
    assert "Configura OPENCODE_BASE_URL y OPENCODE_API_KEY" in capsys.readouterr().err


@pytest.mark.parametrize("stream", [False, True])
async def test_investigator_opencode_sends_reasoning_none_with_tools(stream) -> None:
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        assert request.url.path.endswith("/openai/v1/chat/completions")
        if body.get("tools") and body.get("reasoning_effort") != "none":
            return httpx.Response(400, json={"error": {"message": "tools require reasoning_effort=none"}})
        if body.get("stream"):
            chunks = [
                {"choices": [{"delta": {"content": "Hola"}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ]
            return httpx.Response(
                200, text="".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n",
                headers={"Content-Type": "text/event-stream"},
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": "Hola"}, "finish_reason": "stop"}]})

    module = runpy.run_path(str(EXAMPLE / "agent.py"))
    agent = module["root_agent"]
    assert agent.generation_config == module["GENERATION_CONFIG"]
    agent.generation_config = {"reasoning_effort": "none"}  # Configuración del caso, no del usuario.
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        async with ModelRouter(
            provider="opencode",
            model="deepseek-v4.1-flash", base_url="https://example.invalid/go",
            api_key="offline", api="chat_completions", http_client=client,
        ) as model:
            agent.model = model
            events = [event async for event in Runner(agent=agent).run_async("user", "session", "Hola", stream=stream)]
    assert events[-1].status == "completed" and events[-1].content == "Hola"
    assert requests and all(body["reasoning_effort"] == "none" and body["tools"] for body in requests)


def test_cli_creates_directly_executable_agents_and_has_no_web(tmp_path: Path) -> None:
    assert "web" not in cli.commands and {"run", "a2a", "mcp"} <= cli.commands.keys()
    target = tmp_path / "new_agent"
    result = CliRunner().invoke(cli, ["create", str(target)])
    assert result.exit_code == 0, result.output
    assert "python" in result.output and "agent.py" in result.output
    (target / ".env").write_text(
        "OPENCODE_BASE_URL=https://example.invalid/go\n"
        "OPENCODE_API_KEY=offline\nOPENCODE_MODEL=offline\n", encoding="utf-8",
    )
    env = dict(os.environ)
    env.pop("OPENCODE_BASE_URL", None)
    env.pop("OPENCODE_API_KEY", None)
    imported = subprocess.run(
        [sys.executable, "-c", "import os, sys; from nas.adk.cli.loader import load_agent; "
         "load_agent(sys.argv[1]); assert os.getenv('OPENCODE_API_KEY') == 'offline'",
         str(target / "agent.py")],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert imported.returncode == 0, imported.stderr
    assert not imported.stdout  # El loader no debe entrar en el bloque __main__.
    executed = subprocess.run(
        [sys.executable, str(target / "agent.py")], input="/salir\n", cwd=tmp_path,
        encoding="utf-8", capture_output=True, env=env, timeout=60,
    )
    assert executed.returncode == 0, executed.stderr
    assert "Chat:" in executed.stdout and "Tú >" in executed.stdout


async def test_conversational_chat_tools_history_reset_and_no_duplicate_stream(monkeypatch, capsys) -> None:
    agent = runpy.run_path(str(EXAMPLE / "agent.py"))["root_agent"]
    model = ScriptedLlm(responses=[
        LlmResponse(tool_calls=[call("investigar", {"tema": "energía solar"})]),
        LlmResponse(content="Respuesta única."),
        LlmResponse(content="Resumen anterior."),
        LlmResponse(content="Sesión nueva."),
    ])
    agent.model = model
    messages = iter(["Investiga energía solar", "Resume el informe anterior", "/reset", "Hola", "/salir"])
    monkeypatch.setattr(Console, "input", lambda *args, **kwargs: next(messages))
    with patch.object(Agent, "aclose", new_callable=AsyncMock) as close:
        await run_chat(agent)
        close.assert_awaited_once()
    output = capsys.readouterr().out
    assert "Tool: investigar" in output and "=== INFORME NAS ===" in output
    assert output.count("Respuesta única.") == 1 and output.count("Resumen anterior.") == 1
    assert [message["content"] for message in model.requests[2].messages if message["role"] == "user"] == [
        "Investiga energía solar", "Resume el informe anterior",
    ]
    assert [message["content"] for message in model.requests[3].messages if message["role"] == "user"] == ["Hola"]


async def test_chat_reports_error_and_allows_next_message(monkeypatch, capsys) -> None:
    model = ScriptedLlm(responses=[
        LlmResponse(content="incompleta", finish_reason="length"), LlmResponse(content="respuesta nueva"),
    ])
    agent = Agent(name="test", model=model)
    messages = iter(["primero", "segundo", "/salir"])
    monkeypatch.setattr(Console, "input", lambda *args, **kwargs: next(messages))
    await run_chat(agent)
    output = capsys.readouterr()
    assert "[error]" in output.err and output.out.count("respuesta nueva") == 1
    assert model.requests[-1].messages == [{"role": "user", "content": "segundo"}]


async def test_rich_commands_and_tool_previews_do_not_change_model_history(monkeypatch, capsys) -> None:
    def detail() -> str:
        return "visible\n" + "line\n" * 6 + "TAIL_MARKER\n[red]literal[/red]"

    model = ScriptedLlm(responses=[
        LlmResponse(tool_calls=[call("detail", {})]), LlmResponse(content="**Primera** respuesta."),
        LlmResponse(tool_calls=[call("detail", {})]), LlmResponse(content="**Segunda** respuesta."),
    ])
    messages = iter(["/help", "/desconocido", "primero", "/tools", "segundo", "/salir"])
    monkeypatch.setattr(Console, "input", lambda *args, **kwargs: next(messages))
    await run_chat(Agent(name="[red]literal[/red]", model=model, tools=[detail]))
    output = capsys.readouterr().out
    assert "Comandos" in output and "Comando desconocido" in output
    assert "Resultado abreviado" in output and "Tools: detalles completos" in output
    assert output.count("TAIL_MARKER") == 1 and "[red]literal[/red]" in output
    assert "**Primera**" not in output and "Primera respuesta." in output
    assert "\x1b" not in output  # Pipes/captura: sin animaciones ni escapes ANSI.
    assert [message["content"] for message in model.requests[2].messages if message["role"] == "user"] == [
        "primero", "segundo",
    ]


async def test_rich_live_streams_markdown_and_cleans_up_after_errors(monkeypatch) -> None:
    output, errors = io.StringIO(), io.StringIO()
    console = Console(file=output, force_terminal=True, width=60)
    error_console = Console(file=errors, force_terminal=False, width=60)
    lives, updates = [], []

    class ObservedLive(Live):
        def __enter__(self):
            lives.append(self)
            return super().__enter__()

        def update(self, renderable, *, refresh=False):
            updates.append(renderable)
            super().update(renderable, refresh=refresh)

    monkeypatch.setitem(run_chat.__globals__, "Console", lambda **kwargs: error_console if kwargs.get("stderr") else console)
    monkeypatch.setitem(_chat_turn.__globals__, "Live", ObservedLive)
    messages = iter(["primero", "segundo", "/salir"])
    monkeypatch.setattr(Console, "input", lambda *args, **kwargs: next(messages))
    model = ScriptedLlm(responses=[
        LlmResponse(content="**Parcial**", finish_reason="length"),
        LlmResponse(content="```python\nprint(1)\n```", usage={"prompt_tokens": 10, "completion_tokens": 5}),
    ])
    await run_chat(Agent(name="test", model=model))
    assert len(lives) == 2 and all(not live.is_started for live in lives)
    assert any(isinstance(renderable, Group) and any(isinstance(child, Markdown) for child in renderable.renderables) for renderable in updates)
    assert "[error]" in errors.getvalue() and "Parcial" in output.getvalue()
    assert "tokens: 10 entrada / 5 salida" in output.getvalue() and "\x1b[" in output.getvalue()


async def test_rich_cancellation_keeps_partial_text_and_releases_session() -> None:
    ready = asyncio.Event()

    class BlockingModel(ScriptedLlm):
        async def generate_stream_async(self, request):
            yield LlmResponse(content="Texto parcial", type="delta")
            ready.set()
            await asyncio.Event().wait()

    output = io.StringIO()
    service = InMemorySessionService()
    runner = Runner(agent=Agent(name="test", model=BlockingModel()), session_service=service)
    task = asyncio.create_task(_chat_turn(runner, "session", "Hola", Console(file=output), False))
    try:
        await asyncio.wait_for(ready.wait(), timeout=5)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    service.acquire_session("session")  # El renderer cerró el generador y liberó el guard.
    service.release_session("session")
    assert output.getvalue().count("Texto parcial") == 1
    assert "Respuesta incompleta" in output.getvalue()


@pytest.mark.parametrize("error", [EOFError, KeyboardInterrupt])
async def test_chat_closes_on_eof_or_interrupted_input(monkeypatch, error) -> None:
    def interrupt(*args, **kwargs):
        raise error()

    monkeypatch.setattr(Console, "input", interrupt)
    with patch.object(Agent, "aclose", new_callable=AsyncMock) as close:
        await run_chat(Agent(name="test", model=ScriptedLlm()))
        close.assert_awaited_once()
