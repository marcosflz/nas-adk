from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from textual.theme import Theme

if TYPE_CHECKING:
    from nas.adk.tui.app import AgentApp


@dataclass(frozen=True)
class Command:
    """One local action shared by the menu, slash input and an optional shortcut."""

    name: str
    title: str
    callback: Callable[[AgentApp, str], Awaitable[None] | None]
    help: str = ""
    key: str = ""
    accepts_args: bool = False

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", self.name):
            raise ValueError("Command names must use lowercase letters, numbers, _ or -")
        if not self.title.strip() or not callable(self.callback):
            raise ValueError("Commands need a title and a callable callback")


@dataclass(frozen=True)
class TuiConfig:
    """Describe the UI before creating an app; agents and connections stay in the ADK.

    ``keymap`` uses Textual binding IDs: menu, send, newline, new, agents, tools,
    cancel and quit. Custom commands use ``Command.key``.
    ``css_path`` accepts a native Textual CSS file.
    """

    title: str = "NAS"
    subtitle: str = "Agentes · ADK"
    welcome: str = "¿Qué quieres hacer?"
    prompt: str = "Escribe un mensaje o /help"
    theme: str | Theme = field(default_factory=lambda: Theme(
        name="nas", background="#161616", surface="#222222", panel="#1c1c1c",
        foreground="#e5e5e5", primary="#e4b88a", secondary="#a3b8ca", accent="#e4b88a",
        success="#9bbf9b", warning="#e4b88a", error="#e59898",
    ))
    content_width: int = 120
    sidebar: bool = False
    css_path: str | Path | None = None
    commands: tuple[Command, ...] = ()
    keymap: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.title.strip() or self.content_width < 40:
            raise ValueError("Configure a title and a content_width of at least 40")
        names = [command.name for command in self.commands]
        if len(names) != len(set(names)):
            raise ValueError("Command names must be unique")
        if any(not key.strip() for key in self.keymap.values()):
            raise ValueError("Key bindings must not be blank")
