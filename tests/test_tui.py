from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("textual")

from textual.command import CommandInput, CommandPalette
from textual.widgets import TextArea
from click.testing import CliRunner

from nas.adk import Agent
from nas.adk.cli.main import main as cli
from nas.adk.core.events.event import Event
from nas.adk.models.base_llm import LlmResponse
from nas.adk.sessions import InMemorySessionService, SQLiteSessionService
from nas.adk.tui import AgentApp, ChatMessage, Command, TuiConfig
from nas.adk.tui.widgets import ToolCall, ToolTurn
from tests.test_runner_contracts import ScriptedLlm, call
from tests.fakes import FakeLlm


@pytest.mark.parametrize("kwargs", [
    {}, {"agents": {}}, {"agents": {"bad": object()}},
    {"agent": Agent(name="test", model=FakeLlm()), "default_agent": "missing"},
    {"agent": Agent(name="test", model=FakeLlm()), "user_id": ""},
    {"agent": Agent(name="test", model=FakeLlm()), "runner_options": {"agent": object()}},
    {"agent": Agent(name="test", model=FakeLlm()), "tool_policy": {"tool": "typo"}},
])
def test_tui_validates_configuration(kwargs) -> None:
    with pytest.raises(ValueError):
        AgentApp(**kwargs)


async def test_tui_stream_tools_history_agent_switch_and_reset() -> None:
    def lookup(topic: str) -> str:
        return f"[red]literal[/red] {topic}"

    first = ScriptedLlm(responses=[
        LlmResponse(tool_calls=[call("lookup", {"topic": "solar"})]),
        LlmResponse(content="**Respuesta única**."),
        LlmResponse(content="Seguimiento."),
        LlmResponse(content="Nueva sesión."),
    ])
    second = ScriptedLlm(responses=[LlmResponse(content="Otro agente.")])
    app = AgentApp(
        agents={"first": Agent(name="first", model=first, tools=[lookup]),
                "second": Agent(name="second", model=second)},
        tool_policy={"lookup": "allow"},
    )
    async with app.run_test(size=(100, 35)) as pilot:
        app.composer.load_text("Hola\nsolar")
        await pilot.press("f5")
        await asyncio.wait_for(app.workers.wait_for_complete(), 5)
        messages = list(app.chat.query(ChatMessage))
        assert [message.message_text for message in messages] == ["Hola\nsolar", "**Respuesta única**."]
        group = app.chat.query_one(ToolTurn)
        assert group.collapsed and group.finished and len(group.calls) == 1
        row = group.query_one(ToolCall)
        assert row.state == "completed" and "[red]literal[/red]" in str(row.result.content)
        assert not app.activity.children and not app.chat.query(".thinking")
        assert "Listo" in str(app._status.content)
        assert not app._busy and app.composer.text == ""
        original_session = app.session_id
        app.composer.load_text("Seguimiento")
        await pilot.press("f5")
        await asyncio.wait_for(app.workers.wait_for_complete(), 5)
        assert [m["content"] for m in first.requests[-1].messages if m["role"] == "user"] == [
            "Hola\nsolar", "Seguimiento",
        ]
        await app.select_agent("second")
        await pilot.pause()
        app.composer.load_text("Otro")
        await pilot.press("f5")
        await asyncio.wait_for(app.workers.wait_for_complete(), 5)
        assert second.requests[0].messages == [{"role": "user", "content": "Otro"}]
        await app.select_agent("first")
        await pilot.pause()
        assert app.session_id == original_session
        assert any(m.message_text == "Seguimiento." for m in app.chat.query(ChatMessage))
        assert app.chat.query_one(ToolTurn).finished
        await pilot.press("ctrl+n")
        assert app.session_id != original_session and not app.chat.query(ChatMessage)
        app.composer.load_text("Nueva")
        await pilot.press("f5")
        await asyncio.wait_for(app.workers.wait_for_complete(), 5)
        assert first.requests[-1].messages == [{"role": "user", "content": "Nueva"}]


@pytest.mark.parametrize("choice", ["allow", "deny", "escape", "enter"])
async def test_tui_tool_approval(choice: str) -> None:
    calls = []

    def write_file(path: str) -> str:
        calls.append(path)
        return "saved"

    model = ScriptedLlm(responses=[
        LlmResponse(tool_calls=[call("write_file", {"path": "example.txt"})]),
        LlmResponse(content="Finished"),
    ])
    app = AgentApp(Agent(name="test", model=model, tools=[write_file]))
    async with app.run_test(size=(100, 35)) as pilot:
        app.composer.load_text("Write")
        await pilot.press("f5")
        await pilot.pause()
        assert app.screen.query("#deny") and not calls
        if choice in {"escape", "enter"}:
            await pilot.press(choice)
        else:
            await pilot.click(f"#{choice}")
        await asyncio.wait_for(app.workers.wait_for_complete(), 5)
        assert calls == (["example.txt"] if choice == "allow" else [])
        assert len(app.screen_stack) == 1
        if choice != "allow":
            assert "Tool denied" in model.requests[-1].messages[-1]["content"]
        row = app.chat.query_one(ToolCall)
        assert row.state == ("completed" if choice == "allow" else "error")
        assert row.has_class("failed") == (choice != "allow")


async def test_tui_cancel_preserves_partial_and_can_run_again() -> None:
    started = asyncio.Event()

    class BlockingLlm(ScriptedLlm):
        async def generate_stream_async(self, request):
            if not started.is_set():
                yield LlmResponse(content="Texto parcial", type="delta")
                started.set()
                await asyncio.Event().wait()
            else:
                yield LlmResponse(content="Después de cancelar")

    service = InMemorySessionService()
    app = AgentApp(Agent(name="test", model=BlockingLlm()), session_service=service)
    async with app.run_test(size=(100, 35)) as pilot:
        app.composer.load_text("Primero")
        await pilot.press("f5")
        await asyncio.wait_for(started.wait(), 5)
        assert app._busy
        app.composer.load_text("Después")  # The editor remains responsive during execution.
        await pilot.press("escape")
        await asyncio.wait_for(app.workers.wait_for_complete(), 5)
        assert [m.message_text for m in app.chat.query(ChatMessage)].count("Texto parcial") == 1
        assert any(m.message_text == "Ejecución cancelada." for m in app.chat.query(ChatMessage))
        assert not app.chat.query(".thinking")
        service.acquire_session(app.session_id)
        service.release_session(app.session_id)
        await pilot.press("f5")
        await asyncio.wait_for(app.workers.wait_for_complete(), 5)
        assert any(m.message_text == "Después de cancelar" for m in app.chat.query(ChatMessage))


async def test_tui_subagent_events_are_visible_once() -> None:
    child = Agent(name="child", model=ScriptedLlm(responses=[LlmResponse(content="Child answer")]))
    root = Agent(name="root", sub_agents=[child], model=ScriptedLlm(responses=[
        LlmResponse(tool_calls=[call("request_task_child", {"task": "Research"})]),
        LlmResponse(content="Root answer"),
    ]))
    app = AgentApp(root, tool_policy={"request_task_child": "allow"})
    async with app.run_test(size=(100, 35)) as pilot:
        app.composer.load_text("Research")
        await pilot.press("f5")
        await asyncio.wait_for(app.workers.wait_for_complete(), 5)
        assert [m.message_text for m in app.chat.query(ChatMessage)] == [
            "Research", "Child answer", "Root answer",
        ]


@pytest.mark.parametrize("sidebar", [False, True])
@pytest.mark.parametrize("status", ["cancelled", "budget_exhausted"])
async def test_tui_grouped_activity_progress_and_interrupted_calls(sidebar: bool, status: str) -> None:
    app = AgentApp(Agent(name="test", model=ScriptedLlm()), config=TuiConfig(sidebar=sidebar))
    async with app.run_test(size=(100, 35)) as pilot:
        app.chat.running = True
        await app.show_event(Event(role="user", content="Busca", invocation_id="root"))
        assert app.chat.query(".thinking")
        for kind in ("reasoning", "analysis", "thought", "stream"):
            await app.show_event(Event(type=kind, invocation_id="root"))
        assert len(app.chat.query(ChatMessage)) == 1 and app.chat.query(".thinking")
        await app.show_event(Event(role="assistant", invocation_id="root", tool_calls=[
            call("lookup", {"query": "[red]literal[/red]"}).model_dump(),
            call("wait", "invalid JSON", "pending").model_dump(),
        ]))
        target = app.activity if sidebar else app.chat
        group = target.query_one(ToolTurn)
        assert group.collapsed and not group.finished and not app.chat.query(".thinking")
        await app.show_event(Event(
            role="assistant", invocation_id="child", parent_invocation_id="root",
            tool_calls=[call("lookup", {"query": "child"}).model_dump()],
        ))
        await app.show_event(Event(
            role="tool", author="lookup", invocation_id="child", parent_invocation_id="root",
            tool_call_id="call1", content='{"error": "failed"}',
        ))
        await app.show_event(Event(type="status", status="completed", invocation_id="child",
                                   parent_invocation_id="root"))
        assert not group.finished and len(group.calls) == 3
        assert group.calls[("child", "call1")].state == "error"
        result = Event(
            role="tool", author="lookup", invocation_id="root", tool_call_id="call1",
            content="\x1b[31m[red]literal[/red]" + "x" * 16000,
            metrics={"tool_seconds": 0.5},
        )
        await app.show_event(result)
        row = group.calls[("root", "call1")]
        assert row.state == "completed" and "0.5 s" in row.title
        assert "[red]literal[/red]" in str(row.result.content)
        assert "\x1b" not in str(row.result.content) and result.id in str(row.result.content)
        assert len(str(row.result.content)) < 16200
        await app.execute_command("tools")
        await pilot.pause()
        if sidebar:
            assert not app.query_one("#activity-panel").display and not app.chat.query(ToolTurn)
        else:
            assert not group.collapsed and not app.query_one("#activity-panel").display
        await app.execute_command("tools")
        await app.show_event(Event(type="stream", role="assistant", invocation_id="root",
                                   content="Respuesta parcial"))
        await app.show_event(Event(type="status", status=status, final=True, invocation_id="root",
                                   metrics={"total_seconds": 1.2}))
        assert group.finished and group.has_class("failed")
        pending = group.calls[("root", "pending")]
        assert pending.state == ("cancelled" if status == "cancelled" else "interrupted")
        assert "No se recibió" in str(pending.result.content)
        assert [m.message_text for m in app.chat.query(ChatMessage)].count("Respuesta parcial") == 1
        assert not app.chat.query(".thinking")
        app.chat.running = False
        await app.action_new_session()
        assert not app.chat.children and not app.activity.children


async def test_tui_sqlite_replay_and_owned_resource_cleanup(tmp_path: Path) -> None:
    db = tmp_path / "sessions.db"
    agent = Agent(name="test", model=ScriptedLlm(responses=[LlmResponse(content="Persisted")]))
    with patch.object(Agent, "aclose", new_callable=AsyncMock) as close:
        app = AgentApp(agent, session_db=db, session_id="session")
        async with app.run_test() as pilot:
            app.composer.load_text("Hola")
            await pilot.press("f5")
            await asyncio.wait_for(app.workers.wait_for_complete(), 5)
        close.assert_awaited_once()
    service = SQLiteSessionService(str(db))
    try:
        app = AgentApp(agent, session_service=service, session_id="session", close_agents=False)
        async with app.run_test():
            assert [m.message_text for m in app.chat.query(ChatMessage)] == ["Hola", "Persisted"]
        assert service.get_session("session") is not None  # Borrowed service remains open.
    finally:
        service.close()


async def test_tui_demo_is_offline_and_multiline_editor() -> None:
    app = AgentApp(Agent(
        name="NAS-Agent",
        model=ScriptedLlm(responses=[LlmResponse(content="Datos FICTICIOS de demostración.")]),
    ))
    async with app.run_test(size=(100, 35)) as pilot:
        await pilot.press("h", "o", "l", "a", "shift+enter", "m", "u", "n", "d", "o")
        assert app.query_one(TextArea).text == "hola\nmundo"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), 5)
        assert any("Datos FICTICIOS" in m.message_text for m in app.chat.query(ChatMessage))


async def test_tui_shutdown_cancels_active_run_before_closing_agents() -> None:
    started = asyncio.Event()
    stopped = asyncio.Event()

    class BlockingLlm(ScriptedLlm):
        async def generate_stream_async(self, request):
            try:
                yield LlmResponse(content="Partial", type="delta")
                started.set()
                await asyncio.Event().wait()
            finally:
                stopped.set()

    service = InMemorySessionService()
    app = AgentApp(Agent(name="test", model=BlockingLlm()), session_service=service)

    async def close() -> None:
        assert stopped.is_set()

    with patch.object(Agent, "aclose", side_effect=close) as close_agent:
        async with app.run_test() as pilot:
            app.composer.load_text("Start")
            await pilot.press("f5")
            await asyncio.wait_for(started.wait(), 5)
        close_agent.assert_awaited_once()
    service.acquire_session(app.session_id)
    service.release_session(app.session_id)


def test_tui_cli_and_optional_dependency_boundary(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-B", "-c", "import sys; from nas.adk import Agent, Runner; "
         "from nas.adk.cli.main import main; assert 'textual' not in sys.modules"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    source = tmp_path / "agent.py"
    source.write_text(
        "from nas.adk import Agent\nfrom tests.fakes import FakeLlm\n"
        "root_agent = Agent(name='test', model=FakeLlm())\n"
    )
    with patch.object(AgentApp, "run") as run:
        result = CliRunner().invoke(cli, ["tui", str(source)])
    assert result.exit_code == 0, result.output
    run.assert_called_once()


@pytest.mark.parametrize("width", [60, 100, 140])
async def test_tui_compact_layout_menu_and_slash_commands(width: int) -> None:
    model = ScriptedLlm()
    app = AgentApp(agents={
        "first": Agent(name="first", model=model),
        "second": Agent(name="second", model=model),
    })
    async with app.run_test(size=(width, 24)) as pilot:
        panel = app.query_one("#activity-panel")
        assert not panel.display and app.query_one("#welcome").display
        assert app.query_one("#conversation").size.width >= min(width, app.ui.content_width) - 6
        assert app.composer.region.bottom < 24
        assert app.query_one("#bottom-bar").region.bottom <= 24
        app.composer.load_text("Un borrador")
        await pilot.press("ctrl+p")
        assert isinstance(app.screen, CommandPalette)
        app.screen.query_one(CommandInput).value = "/tools"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert len(app.screen_stack) == 1 and not panel.display
        assert app.composer.text == "Un borrador" and not model.requests
        app.composer.load_text("/agents second")
        await pilot.press("enter")
        assert app.selected_agent == "second" and not app.composer.text
        app.composer.load_text("/agents")
        await pilot.press("enter")
        assert app.screen.id == "agents-menu"
        app.screen.query_one(CommandInput).value = "first"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app.selected_agent == "first"
        original = app.session_id
        app.composer.load_text("/new")
        await pilot.press("enter")
        assert app.session_id != original and not model.requests
        app.composer.load_text("/no-existe")
        await pilot.press("enter")
        assert app.composer.text == "/no-existe" and not model.requests
        assert "desconocido" in str(app._status.content)
        app.composer.load_text("/new argumento")
        await pilot.press("enter")
        assert app.composer.text == "/new argumento" and "no acepta" in str(app._status.content)
        app.composer.load_text("/help")
        await pilot.press("enter")
        assert isinstance(app.screen, CommandPalette)
        await pilot.press("escape")
        assert len(app.screen_stack) == 1 and app.focused is app.composer


async def test_tui_declarative_config_custom_commands_keymap_and_css(tmp_path: Path) -> None:
    calls = []

    async def review(app: AgentApp, args: str) -> None:
        calls.append((app.selected_agent, args))

    css = tmp_path / "ui.tcss"
    css.write_text("#brand { color: #abcdef; }", encoding="utf-8")
    ui = TuiConfig(
        title="Mi terminal", subtitle="Proyecto", welcome="Bienvenido", prompt="Pregunta…",
        theme="nord", content_width=80, sidebar=True, css_path=css,
        keymap={"menu": "ctrl+k", "send": "f4", "newline": "ctrl+j"},
        commands=(Command("review", "Revisar", review, key="f6", accepts_args=True),),
    )
    model = ScriptedLlm(responses=[LlmResponse(content="Respuesta")])
    app = AgentApp(Agent(name="test", model=model), config=ui)
    async with app.run_test(size=(120, 30)) as pilot:
        assert app.title == "Mi terminal" and app.theme == "nord"
        assert app.query_one("#workspace").region.width == 80
        assert app.query_one("#activity-panel").display
        assert app.query_one("#brand").styles.color.hex == "#ABCDEF"
        assert "Bienvenido" in str(app.query_one("#welcome").content)
        assert "f4 enviar" in str(app.query_one("#shortcuts").content)
        await pilot.press("ctrl+b")
        assert not app.query_one("#activity-panel").display
        await pilot.press("ctrl+b")
        assert app.query_one("#activity-panel").display
        app.composer.load_text("/review\tarchivo.py\notro.py")
        await pilot.press("f4")
        assert calls == [("test", "archivo.py\notro.py")] and not model.requests
        await pilot.press("f6")
        assert calls[-1] == ("test", "")
        await pilot.press("ctrl+k")
        assert isinstance(app.screen, CommandPalette)
        app.screen.query_one(CommandInput).value = "/review"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert len(calls) == 3 and len(app.screen_stack) == 1
        app.composer.load_text("Hola")
        await pilot.press("f4")
        await asyncio.wait_for(app.workers.wait_for_complete(), 5)
        assert model.requests[0].messages == [{"role": "user", "content": "Hola"}]
        await pilot.press("ctrl+p")
        assert len(app.screen_stack) == 1  # Old palette shortcut was replaced, not added.


async def test_tui_menu_during_stream_preserves_run_and_draft() -> None:
    started, finish = asyncio.Event(), asyncio.Event()

    class BlockingLlm(ScriptedLlm):
        async def generate_stream_async(self, request):
            started.set()
            await finish.wait()
            yield LlmResponse(content="Terminado")

    app = AgentApp(Agent(name="test", model=BlockingLlm()))
    async with app.run_test() as pilot:
        app.composer.load_text("Ejecuta")
        await pilot.press("enter")
        await asyncio.wait_for(started.wait(), 5)
        original = app.session_id
        assert not await app.execute_command("new") and app.session_id == original
        app.composer.load_text("Siguiente mensaje")
        await pilot.press("ctrl+p")
        assert isinstance(app.screen, CommandPalette)
        await app.show_event(Event(role="assistant", content="Actividad en segundo plano"))
        assert app._busy
        await pilot.press("escape")
        assert app._busy and app.composer.text == "Siguiente mensaje"
        finish.set()
        await asyncio.wait_for(app.workers.wait_for_complete(), 5)
        assert any(m.message_text == "Terminado" for m in app.chat.query(ChatMessage))


def test_tui_command_configuration_validation() -> None:
    def callback(app: AgentApp, args: str) -> None:
        pass

    with pytest.raises(ValueError, match="lowercase"):
        Command("invalid name", "Inválido", callback)
    command = Command("custom", "Propio", callback)
    with pytest.raises(ValueError, match="unique"):
        TuiConfig(commands=(command, command))
    with pytest.raises(ValueError, match="Reserved"):
        AgentApp(Agent(name="test", model=FakeLlm()), config=TuiConfig(commands=(
            Command("new", "Reservado", callback),
        )))
