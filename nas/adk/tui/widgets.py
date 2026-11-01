from __future__ import annotations

import json
from typing import Any

from rich.markdown import Markdown
from rich.text import Text
from textual.containers import VerticalScroll
from textual.widgets import Collapsible, Static

from nas.adk.core.events.event import Event


def plain(value: str) -> str:
    """Discard terminal control sequences before rendering untrusted text."""
    return Text.from_ansi(value).plain


class ChatMessage(Static):
    DEFAULT_CSS = """
    ChatMessage { height: auto; border: none; background: transparent;
        margin: 0 0 1 0; padding: 0; }
    ChatMessage.user { border-left: tall $secondary; background: $panel; padding: 1 2; }
    ChatMessage.status { color: $warning; }
    """

    def __init__(self, author: str, content: str = "", *, role: str = "assistant") -> None:
        super().__init__(classes=role, markup=False)
        self.border_title = plain(author)
        self.message_text = content
        self._deltas: list[str] = []
        self._dirty = True
        self.flush()

    def append_delta(self, content: str) -> None:
        self._deltas.append(content)
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        self.message_text += "".join(self._deltas)
        self._deltas.clear()
        content = plain(self.message_text)
        self.update(Markdown(content) if self.has_class("assistant") else Text(content))
        self._dirty = False

    def finish(self, content: str | None = None) -> None:
        if content is not None:
            self.message_text = content
            self._deltas.clear()
        self._dirty = True
        self.flush()


class ToolCall(Collapsible):
    """One expandable call, pairing its arguments with its eventual result."""

    DEFAULT_CSS = """
    ToolCall { border: none; background: transparent; background-tint: transparent;
        padding: 0; margin: 0; }
    ToolCall Static { height: auto; color: $text-muted; margin: 1 0; }
    ToolCall.failed > CollapsibleTitle { color: $error; }
    """

    def __init__(self, name: str, arguments: Any) -> None:
        raw = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
        try:
            args = json.loads(raw)
        except ValueError:
            args = None
        detail = ""
        if isinstance(args, dict):
            detail = next((value for value in args.values() if isinstance(value, str)), "")
        self.label = plain(f"{name}  {detail}").replace("\n", " ")[:80].strip()
        self.state = "running"
        self.result = Static("Esperando resultado…", markup=False, classes="tool-output")
        super().__init__(
            Static(Text(f"Entrada\n{plain(raw)}"), classes="tool-input"), self.result,
            title=f"· {self.label}", collapsed=True,
            collapsed_symbol="›", expanded_symbol="⌄",
        )

    def finish(self, event: Event) -> None:
        content = plain(event.content or "")
        try:
            data = json.loads(content)
        except ValueError:
            data = None
        error = event.status == "error" or (isinstance(data, dict) and "error" in data)
        self.state = "error" if error else "completed"
        seconds = event.metrics.get("tool_seconds")
        suffix = f" · {seconds:.1f} s" if isinstance(seconds, (int, float)) else ""
        self.title = f"{'×' if error else '✓'} {self.label}{suffix}{' · Error' if error else ''}"
        self.set_class(error, "failed")
        # ponytail: UI results capped at 16000 chars; the session retains the full event.
        preview = content[:16000]
        if preview != content:
            preview += f"\n[Abreviado; resultado completo: event:{event.id}]"
        self.result.update(Text(f"Resultado\n{preview}"))


class ToolTurn(Collapsible):
    """Compact activity for a user turn, including delegated invocations."""

    DEFAULT_CSS = """
    ToolTurn { border: none; background: transparent; background-tint: transparent;
        padding: 0; margin: 0 0 1 0; }
    ToolTurn CollapsibleTitle { width: 1fr; height: 1; padding: 0;
        color: $text-muted; text-style: none; text-wrap: nowrap; text-overflow: ellipsis; }
    ToolTurn CollapsibleTitle:hover { color: $foreground; background: $panel; }
    ToolTurn CollapsibleTitle:focus { color: $primary; background: $surface; text-style: bold; }
    ToolTurn Collapsible > Contents, ToolTurn > Contents { padding: 0 0 0 2; }
    ToolTurn.failed > CollapsibleTitle { color: $error; }
    """

    def __init__(self) -> None:
        super().__init__(title="Usando herramientas…", collapsed=True,
                         collapsed_symbol="›", expanded_symbol="⌄")
        self.calls: dict[tuple[str, str], ToolCall] = {}
        self.finished = False

    async def add_call(self, key: tuple[str, str], name: str, arguments: Any) -> ToolCall:
        if key not in self.calls:
            call = ToolCall(name, arguments)
            self.calls[key] = call
            await self.query_one(Collapsible.Contents).mount(call)
        return self.calls[key]

    async def show_event(self, event: Event) -> None:
        for call in event.tool_calls:
            function = call.get("function", {})
            await self.add_call(
                (event.invocation_id, call["id"]),
                function.get("name", "Herramienta"), function.get("arguments", "{}"),
            )
        if event.role == "tool":
            row = await self.add_call(
                (event.invocation_id, event.tool_call_id), event.author, "{}",
            )
            row.finish(event)
        if not self.finished:
            done = sum(call.state != "running" for call in self.calls.values())
            total = len(self.calls)
            summary = f"{total} {'llamada' if total == 1 else 'llamadas'}"
            active = next((call.label for call in self.calls.values() if call.state == "running"), "")
            self.title = (
                f"Herramientas · {done}/{total} · {active}" if done < total
                else f"{summary} · Preparando respuesta…"
            )
            if event.type == "stream" and event.content:
                self.title = f"{summary} · Respondiendo…"
        if not event.parent_invocation_id and (event.final or event.type == "status"):
            self.finish(event.status or ("completed" if event.final else "interrupted"))

    def finish(self, status: str = "interrupted") -> None:
        if self.finished:
            return
        if status == "completed" and any(call.state == "running" for call in self.calls.values()):
            status = "interrupted"
        for call in self.calls.values():
            if call.state == "running":
                call.state = "cancelled" if status == "cancelled" else "interrupted"
                label = "Cancelada" if status == "cancelled" else "Interrumpida"
                call.title = f"– {call.label} · {label}"
                call.result.update("No se recibió un resultado.")
        total = len(self.calls)
        errors = sum(call.state == "error" for call in self.calls.values())
        state = {"completed": "completada" if total == 1 else "completadas",
                 "cancelled": "turno cancelado"}.get(
            status, "turno interrumpido",
        )
        if errors:
            state += f" · {errors} con error"
        self.title = f"{total} {'llamada' if total == 1 else 'llamadas'} · {state}"
        self.set_class(bool(errors), "failed")
        self.finished = True


class ChatView(VerticalScroll):
    """Render canonical ADK events, replacing streamed text with its consolidated reply."""

    DEFAULT_CSS = """
    ChatView { height: 1fr; padding: 0 1 0 0; scrollbar-size: 1 1;
        scrollbar-background: $background; scrollbar-color: $surface-lighten-2;
        scrollbar-color-hover: $text-muted; scrollbar-color-active: $primary; }
    ChatView .thinking { height: 1; margin-bottom: 1; color: $primary; }
    """

    def __init__(self, *, inline_tools: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.inline_tools = inline_tools
        self.running = False
        self._pending: dict[str, ChatMessage] = {}
        self._tool_turn: ToolTurn | None = None
        self._thinking: Static | None = None
        self._spinner_frame = 0

    async def set_progress(self, title: str | None) -> None:
        if self._thinking is not None:
            await self._thinking.remove()
            self._thinking = None
        if title:
            self._thinking = Static(title, markup=False, classes="thinking")
            await self.mount(self._thinking)

    async def show_event(self, event: Event) -> None:
        if event.type.casefold() in {"analysis", "reasoning", "thought"}:
            return
        if event.type == "stream" and not event.content and not event.tool_calls:
            return
        follow = self.is_vertical_scroll_end
        if event.role == "user" and not event.parent_invocation_id:
            self.finish_pending()
            self._tool_turn = None
            await self.set_progress(None)
        elif event.content or event.tool_calls or event.final or event.type == "status":
            await self.set_progress(None)
        if event.type == "stream":
            if event.content:
                message = self._pending.get(event.invocation_id)
                if message is None:
                    message = ChatMessage(event.author)
                    self._pending[event.invocation_id] = message
                    await self.mount(message)
                message.append_delta(event.content)
        elif event.type == "status":
            if not event.parent_invocation_id:
                self.finish_pending(event.status or "interrupted")
            content = event.content or {
                "cancelled": "Ejecución cancelada.",
                "budget_exhausted": "Se alcanzó el límite de ejecución.",
                "error": "La ejecución ha fallado.",
            }.get(event.status or "", event.status or "Estado")
            await self.mount(ChatMessage(
                event.status or "Estado", content, role="status",
            ))
        elif event.role in {"user", "assistant"}:
            message = self._pending.pop(event.invocation_id, None)
            if message is not None:
                message.finish(event.content if event.content else None)
            elif event.content:
                await self.mount(ChatMessage(event.author, event.content, role=event.role))
        if self.inline_tools and (event.tool_calls or event.role == "tool"):
            if self._tool_turn is None:
                self._tool_turn = ToolTurn()
                await self.mount(self._tool_turn)
        if self._tool_turn is not None:
            await self._tool_turn.show_event(event)
        if event.role == "user" and not event.parent_invocation_id and self.running:
            await self.set_progress("Pensando…")
        if follow:
            self.scroll_end(animate=False)

    def flush(self) -> None:
        follow = self.is_vertical_scroll_end
        for message in self._pending.values():
            message.flush()
        if self._thinking is not None:
            frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            self._thinking.update(f"{frames[self._spinner_frame % len(frames)]} Pensando…")
            self._spinner_frame += 1
        if follow:
            self.scroll_end(animate=False)

    def finish_pending(self, status: str = "interrupted") -> None:
        self.flush()
        self._pending.clear()
        if self._tool_turn is not None:
            self._tool_turn.finish(status)

    async def clear(self) -> None:
        self._pending.clear()
        self._tool_turn = self._thinking = None
        await self.remove_children()


class ToolActivity(VerticalScroll):
    """Tool calls and expandable results, including delegated agents."""

    DEFAULT_CSS = """
    ToolActivity { height: 1fr; }
    ToolActivity Collapsible { margin: 0 0 1 0; }
    ToolActivity Static { height: auto; }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._group: ToolTurn | None = None

    async def show_event(self, event: Event) -> None:
        if event.role == "user" and not event.parent_invocation_id:
            self.finish_pending()
            self._group = None
        if (event.tool_calls or event.role == "tool") and self._group is None:
            self._group = ToolTurn()
            await self.mount(self._group)
        if self._group is not None:
            await self._group.show_event(event)

    def finish_pending(self) -> None:
        if self._group is not None:
            self._group.finish()

    async def clear(self) -> None:
        self._group = None
        await self.remove_children()
