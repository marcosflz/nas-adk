"""Textual applications and reusable widgets for NAS agents (extra ``[tui]``)."""

try:
    from nas.adk.tui.app import AgentApp
except ModuleNotFoundError as error:
    if error.name != "textual":
        raise
    raise ModuleNotFoundError(
        'Install the TUI extra: pip install "nas-adk[tui]"', name="textual",
    ) from error

from nas.adk.tui.widgets import ChatMessage, ChatView, ToolActivity
from nas.adk.tui.config import Command, TuiConfig

__all__ = ["AgentApp", "ChatMessage", "ChatView", "Command", "ToolActivity", "TuiConfig"]
