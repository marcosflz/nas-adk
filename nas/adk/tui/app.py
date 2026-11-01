from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Iterable, Mapping
from contextlib import AsyncExitStack, aclosing
from functools import partial
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.command import CommandListItem, CommandPalette, SimpleCommand, SimpleProvider
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.theme import Theme
from textual.widgets import Button, Label, Static, TextArea
from textual.worker import Worker, WorkerCancelled, WorkerFailed

from nas.adk.core.callbacks import BaseCallback, CallbackManager
from nas.adk.core.events.event import Event
from nas.adk.runners import Runner
from nas.adk.sessions import InMemorySessionService, SQLiteSessionService
from nas.adk.tui.config import Command, TuiConfig
from nas.adk.tui.widgets import ChatView, ToolActivity, ToolTurn, plain


class _Approval(ModalScreen[bool]):
    DEFAULT_CSS = """
    _Approval { align: center middle; }
    _Approval > Vertical { width: 80%; max-height: 80%; padding: 1 2;
        background: $surface; border: round $warning; }
    _Approval Static { height: auto; max-height: 12; overflow-y: auto; }
    _Approval Horizontal { height: auto; margin-top: 1; }
    """
    BINDINGS = [Binding("escape", "deny", "Denegar", priority=True)]

    def __init__(self, name: str, args: dict[str, Any]) -> None:
        super().__init__()
        self._name, self._args = name, args

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(Text(plain(f"¿Ejecutar {self._name}?")))
            yield Static(Text(plain(json.dumps(self._args, ensure_ascii=False, indent=2))))
            with Horizontal():
                yield Button("Denegar", id="deny", variant="error")
                yield Button("Permitir una vez", id="allow", variant="success")

    def on_mount(self) -> None:
        self.query_one("#deny", Button).focus()

    @on(Button.Pressed)
    def answer(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "allow")

    def action_deny(self) -> None:
        self.dismiss(False)


class _UiCallback(BaseCallback):
    def __init__(self, app: AgentApp) -> None:
        self._app = app
        self._approval_lock = asyncio.Lock()

    async def on_event(self, event: Event) -> None:
        await self._app.show_event(event)

    async def before_tool_call(self, tool_name: str, args: dict[str, Any]) -> None:
        policy = self._app.tool_policy.get(tool_name, "ask")
        if policy == "allow":
            return
        if policy == "ask":
            # One modal at a time, even when the Runner executes parallel-safe tools.
            async with self._approval_lock:
                screen = _Approval(tool_name, args)
                try:
                    if await self._app.push_screen_wait(screen):
                        return
                finally:
                    if screen in self._app.screen_stack:
                        screen.dismiss(False)
        raise PermissionError(f"Tool denied: {tool_name}")


class _Composer(TextArea):
    BINDINGS = [
        Binding("enter,f5", "app.send", "Enviar", id="send", priority=True),
        Binding("shift+enter,ctrl+enter,ctrl+j", "newline", "Nueva línea",
                id="newline", show=False, priority=True),
    ]

    def action_newline(self) -> None:
        self.replace("\n", *self.selection, maintain_selection_offset=False)


class AgentApp(App[None]):
    """Configurable Textual chat using the ADK Runner, not a second agent runtime.

    Agents may be ADK Agents or Workflows. Models, tools (including connected MCP
    tools), memory and limits retain their ADK APIs. ``runner_options`` forwards
    Runner options except agent/app_name/session_service. Tools ask for approval
    unless explicitly allowed or denied in ``tool_policy``.

    Agents and their model instances are app-owned by default and are closed on exit.
    Use close_agents=False when the caller manages their lifetime.
    """

    DEFAULT_CSS = """
    Screen { background: $background; color: $foreground; }
    #shell { width: 100%; height: 1fr; align-horizontal: center; }
    #workspace { width: 100%; height: 1fr; padding: 1 3 0 3; }
    #brand { height: 1; padding: 0; margin-bottom: 1; color: $foreground;
        text-wrap: nowrap; text-overflow: ellipsis; }
    #body { height: 1fr; }
    #conversation { width: 1fr; height: 1fr; border: none; background: $background; padding: 0; }
    #welcome { height: 1fr; content-align: center middle; color: $text-muted; }
    #activity-panel { width: 32%; min-width: 20; border-left: solid $surface-lighten-2;
        padding: 0 1; }
    #activity-panel Label { height: 1; color: $text-muted; }
    #input-box { height: auto; background: $surface; padding: 1 1 0 1; border: none;
        border-left: tall $primary; margin-top: 1; }
    #composer { height: 3; max-height: 8; border: none; padding: 0 1; background: $surface; }
    #composer:focus { border: none; }
    #composer .text-area--cursor-line { background: transparent; }
    #context { height: 1; padding: 0 1; color: $text-muted; }
    #bottom-bar { height: 1; margin: 1 0; }
    #shortcuts { width: 1fr; height: 1; color: $text-muted;
        text-wrap: nowrap; text-overflow: ellipsis; }
    #status { width: auto; max-width: 45%; height: 1; padding: 0 0 0 2;
        color: $text-muted; text-align: right; text-wrap: nowrap; text-overflow: ellipsis; }
    CommandPalette { background: $background 65%; }
    CommandPalette > #--container { width: 80%; max-width: 80; margin-top: 3;
        background: $surface; border: round $surface-lighten-2; }
    """
    BINDINGS = [
        Binding("ctrl+p", "command_palette", "Menú", id="menu", priority=True),
        Binding("escape", "cancel", "Cancelar", id="cancel"),
        Binding("ctrl+n", "new_session", "Nueva sesión", id="new", show=False, priority=True),
        Binding("ctrl+shift+a", "agents", "Agentes", id="agents", show=False, priority=True),
        Binding("ctrl+b", "toggle_activity", "Tools", id="tools", show=False, priority=True),
        Binding("ctrl+q", "quit", "Salir", id="quit", show=False),
    ]

    def __init__(
        self,
        agent: Any = None,
        *,
        agents: Mapping[str, Any] | None = None,
        default_agent: str | None = None,
        config: TuiConfig | None = None,
        title: str | None = None,
        theme: str | Theme | None = None,
        app_name: str = "tui",
        user_id: str = "local_user",
        session_id: str | None = None,
        session_db: str | Path | None = None,
        session_service: InMemorySessionService | None = None,
        runner_options: Mapping[str, Any] | None = None,
        tool_policy: Mapping[str, Literal["allow", "ask", "deny"]] | None = None,
        close_agents: bool = True,
    ) -> None:
        self.ui = config or TuiConfig()
        super().__init__(css_path=self.ui.css_path)
        if agent is not None and agents is not None:
            raise ValueError("Pass agent or agents, not both")
        self.agents = dict(agents or {})
        if agent is not None:
            self.agents[getattr(agent, "name", "") or "assistant"] = agent
        if not self.agents or any(
            not isinstance(name, str) or not name.strip() or not all(
                callable(getattr(value, method, None))
                for method in ("validate_input", "generate_async")
            )
            for name, value in self.agents.items()
        ):
            raise ValueError("Configure at least one named ADK agent or workflow")
        self.selected_agent = default_agent if default_agent is not None else next(iter(self.agents))
        if self.selected_agent not in self.agents:
            raise ValueError("default_agent must be a configured agent")
        if not app_name.strip() or not user_id.strip() or session_id == "":
            raise ValueError("Application, user and session identities must not be blank")
        if session_db is not None and session_service is not None:
            raise ValueError("Pass session_db or session_service, not both")
        self._options = dict(runner_options or {})
        if self._options.keys() & {"agent", "app_name", "session_service"}:
            raise ValueError("runner_options cannot override agent, app_name or session_service")
        self.tool_policy = dict(tool_policy or {})
        if any(policy not in {"allow", "ask", "deny"} for policy in self.tool_policy.values()):
            raise ValueError("Tool policies must be allow, ask or deny")
        selected_theme = theme if theme is not None else self.ui.theme
        if isinstance(selected_theme, Theme):
            self.register_theme(selected_theme)
            selected_theme = selected_theme.name
        self.title, self.theme = plain(title if title is not None else self.ui.title), selected_theme
        self.set_keymap(dict(self.ui.keymap))
        self.app_name, self.user_id = app_name, user_id
        self.session_service = session_service or InMemorySessionService()
        self._session_db = session_db
        self._close_agents = close_agents
        self._sessions = {name: str(uuid4()) for name in self.agents}
        if session_id is not None:
            self._sessions[self.selected_agent] = session_id
        self._resources = AsyncExitStack()
        self._runners: dict[str, Runner] = {}
        self._worker: Worker[None] | None = None
        self._busy = False
        self._closing = False
        self.chat = ChatView(id="chat", inline_tools=not self.ui.sidebar)
        self.activity = ToolActivity(id="activity")
        self.composer = _Composer(id="composer", tab_behavior="focus", placeholder=plain(self.ui.prompt))
        for command in self.ui.commands:
            if command.key:
                self.composer._bindings.bind(
                    command.key, f"app.command('{command.name}')", command.title,
                    show=False, priority=True,
                )
        self.composer.tooltip = self.ui.prompt
        self._status = Static("", id="status", markup=False)
        self._welcome = Static(Text(plain(f"{self.ui.welcome}\n\n{self.ui.prompt}")), id="welcome")
        self._context_label = Static("", id="context", markup=False)
        self._activity_panel = Vertical(Label("TOOLS Y DELEGACIONES"), self.activity,
                                        id="activity-panel")
        self._commands = {
            command.name: command for command in (
                Command("new", "Nueva sesión", lambda app, _: app.action_new_session(),
                        "Empezar una conversación sin borrar el historial"),
                Command("agents", "Cambiar agente", lambda app, arg: app.action_agents(arg),
                        "Seleccionar agente · /agents nombre", accepts_args=True),
                Command("tools", "Desplegar / plegar herramientas",
                        lambda app, _: app.action_toggle_activity()),
                Command("theme", "Cambiar tema", lambda app, _: app.action_change_theme()),
                Command("help", "Comandos y atajos", lambda app, _: app.action_command_palette()),
                Command("cancel", "Cancelar ejecución", lambda app, _: app.action_cancel()),
                Command("quit", "Salir", lambda app, _: app.action_quit()),
            )
        }
        for command in self.ui.commands:
            if command.name in self._commands:
                raise ValueError(f"Reserved command name: {command.name}")
            self._commands[command.name] = command

    @property
    def session_id(self) -> str:
        return self._sessions[self.selected_agent]

    def compose(self) -> ComposeResult:
        with Vertical(id="shell"):
            with Vertical(id="workspace"):
                brand = Text(self.title, style="bold")
                brand.append(f"   /   {plain(self.ui.subtitle)}", style="dim")
                yield Static(brand, id="brand")
                with Horizontal(id="body"):
                    with Vertical(id="conversation"):
                        yield self._welcome
                        yield self.chat
                    yield self._activity_panel
                with Vertical(id="input-box"):
                    yield self.composer
                    yield self._context_label
                with Horizontal(id="bottom-bar"):
                    keys = self.ui.keymap
                    yield Static(Text(
                        f"{keys.get('send', 'enter').split(',')[0]} enviar   "
                        f"{keys.get('newline', 'shift+enter').split(',')[0]} nueva línea   "
                        f"{keys.get('menu', 'ctrl+p').split(',')[0]} menú   "
                        f"{keys.get('cancel', 'esc').split(',')[0]} cancelar",
                    ), id="shortcuts")
                    yield self._status

    async def on_mount(self) -> None:
        self.query_one("#workspace").styles.max_width = self.ui.content_width
        self._activity_panel.display = self.ui.sidebar
        if self._session_db is not None:
            self.session_service = SQLiteSessionService(str(self._session_db))
            self._resources.callback(self.session_service.close)
        if self._close_agents:
            seen: set[int] = set()
            for agent in self.agents.values():
                if id(agent) not in seen and callable(getattr(agent, "aclose", None)):
                    self._resources.push_async_callback(agent.aclose)
                    close_models = getattr(agent, "aclose_models", None)
                    if callable(close_models):
                        self._resources.push_async_callback(close_models)
                    seen.add(id(agent))
        options = dict(self._options)
        callbacks = options.pop("callbacks", None)
        callbacks = callbacks.callbacks if isinstance(callbacks, CallbackManager) else callbacks
        options["callbacks"] = [*(callbacks or []), _UiCallback(self)]
        self._runners = {
            name: Runner(
                app_name=self.app_name, agent=agent, session_service=self.session_service, **options,
            ) for name, agent in self.agents.items()
        }
        # ponytail: full Markdown repaint at 8 Hz; incremental rendering if long replies become costly.
        self.set_interval(0.125, self.chat.flush)
        await self._load_history()
        self.composer.focus()

    async def _load_history(self) -> None:
        await self.chat.clear()
        await self.activity.clear()
        session = await asyncio.to_thread(
            self.session_service.get_session, self.session_id,
            app_name=self.app_name, user_id=self.user_id,
        )
        if session is not None:
            for event in session.events:
                await self.show_event(event)
        self.chat.finish_pending()
        self.activity.finish_pending()
        model = getattr(self.agents[self.selected_agent], "model", "workflow offline")
        model_name = model if isinstance(model, str) else getattr(model, "model", "modelo")
        self._context_label.update(Text(plain(
            f"{self.selected_agent}  ·  {model_name}  ·  {self.session_id[:8]}",
        )))
        self._status.update("")
        self._update_welcome()

    def _update_welcome(self) -> None:
        empty = not self.chat.children
        self._welcome.display = empty
        self.chat.display = not empty

    async def show_event(self, event: Event) -> None:
        if self._closing or not self.chat.is_mounted:
            return
        await self.chat.show_event(event)
        if self.ui.sidebar:
            await self.activity.show_event(event)
        self._update_welcome()
        if event.final:
            usage = event.metrics.get("total_usage", event.usage)
            status = {"completed": "Listo", "cancelled": "Cancelado"}.get(
                event.status or "completed", "Error",
            )
            stats = f"{status} · {event.metrics.get('total_seconds', 0):.1f} s"
            tokens = usage.get("output_tokens", usage.get("completion_tokens")) if usage else None
            if isinstance(tokens, int):
                stats += f" · {tokens:,} tokens"
            self._status.update(Text(stats))

    @on(TextArea.Changed, "#composer")
    def resize_composer(self) -> None:
        self.composer.styles.height = min(8, max(3, self.composer.document.line_count))

    async def action_send(self) -> None:
        if self._closing or not self.composer.text.strip():
            return
        message = self.composer.text.strip()
        if message.startswith("/"):
            parts = message[1:].split(maxsplit=1)
            name, args = (parts[0], parts[1] if len(parts) > 1 else "") if parts else ("", "")
            if await self.execute_command(name, args):
                self.composer.load_text("")
            return
        if self._busy:
            self._status.update("Ejecución en curso · Esc para cancelar")
            return
        self.composer.load_text("")
        self._busy = True
        self._status.update("Ejecutando… · Esc cancela")
        self._worker = self.run_worker(self._run_turn(message), exit_on_error=False)
        self.composer.focus()

    async def _run_turn(self, message: str) -> None:
        self.chat.running = True
        try:
            async with aclosing(self._runners[self.selected_agent].run_async(
                self.user_id, self.session_id, message, stream=True,
            )) as events:
                async for _ in events:
                    pass  # on_event is the single UI feed, including child invocations.
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self.show_event(Event(type="status", status="error", content=str(error)))
        finally:
            self.chat.running = False
            if not self._closing and self.chat.is_mounted:
                await self.chat.set_progress(None)
                self.chat.finish_pending()
                self.activity.finish_pending()
            self._busy = False

    def action_cancel(self) -> None:
        if isinstance(self.screen, _Approval):
            self.screen.dismiss(False)
        elif self._busy and self._worker is not None:
            self._worker.cancel()

    async def action_new_session(self) -> None:
        if self._busy or self._closing:
            return
        self._sessions[self.selected_agent] = str(uuid4())
        await self._load_history()
        self.composer.focus()

    async def select_agent(self, name: str) -> None:
        if name not in self.agents:
            raise ValueError(f"Unknown agent: {name}")
        if self._busy:
            raise RuntimeError("Cancela la ejecución antes de cambiar de agente")
        if name == self.selected_agent:
            return
        self.selected_agent = name
        await self._load_history()
        self.composer.focus()

    def get_system_commands(self, screen: Screen[Any]) -> Iterable[SystemCommand]:
        for command in self._commands.values():
            if self._busy and command.name in {"new", "agents"}:
                continue
            yield SystemCommand(
                plain(f"/{command.name}  {command.title}"), plain(command.help),
                partial(self.action_command, command.name),
            )

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action in {"command_palette", "new_session", "agents", "toggle_activity"}:
            return len(self.screen_stack) == 1 and (
                not self._busy or action not in {"new_session", "agents"}
            )
        return super().check_action(action, parameters)

    def action_command_palette(self) -> None:
        if len(self.screen_stack) == 1:
            self.push_screen(CommandPalette(placeholder="Buscar comandos…", id="command-menu"))

    async def action_command(self, name: str) -> None:
        await self.execute_command(name)

    async def execute_command(self, name: str, args: str = "") -> bool:
        """Execute a local command. Never forward unknown slash commands to the model."""
        command = self._commands.get(name)
        if command is None:
            self._status.update(Text(f"Comando desconocido: /{plain(name)} · /help"))
            return False
        if args and not command.accepts_args:
            self._status.update(Text(f"/{name} no acepta argumentos"))
            return False
        if self._busy and name in {"new", "agents"}:
            self._status.update("Cancela la ejecución antes de cambiar de sesión o agente")
            return False
        try:
            result = command.callback(self, args)
            if inspect.isawaitable(result):
                await result
        except Exception as error:
            self._status.update(Text(plain(f"/{name}: {error}")))
            return False
        return True

    async def action_agents(self, name: str = "") -> None:
        if self._busy or len(self.screen_stack) != 1:
            return
        if name:
            await self.select_agent(name)
        else:
            commands: list[CommandListItem] = [
                SimpleCommand(plain(agent_name), partial(self.select_agent, agent_name))
                for agent_name in self.agents
            ]
            self.push_screen(CommandPalette(
                # Textual accepts instances here, but ProviderSource only annotates classes.
                providers=[SimpleProvider(self.screen, commands)],  # type: ignore[list-item]
                placeholder="Seleccionar agente…", id="agents-menu",
            ))

    def action_toggle_activity(self) -> None:
        if self.ui.sidebar:
            self._activity_panel.display = not self._activity_panel.display
        else:
            groups = list(self.chat.query(ToolTurn))
            collapsed = any(not group.collapsed for group in groups)
            for group in groups:
                group.collapsed = collapsed
        self.composer.focus()

    async def on_unmount(self) -> None:
        self._closing = True
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker.wait()
            except (WorkerCancelled, WorkerFailed):
                pass
        await self._resources.aclose()
