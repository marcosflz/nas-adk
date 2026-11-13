"""Intelligent UI nativo de nas-adk.

- `ui_tools` / `OpenUiTools` / `FollowupTools`: herramientas que entregan interfaces
  (OpenUI Lang), formularios que esperan a la persona y preguntas de seguimiento.
- `TextCallRecovery`: envuelve un modelo y convierte en llamadas reales las de
  presentación que escribe como texto, para que el usuario no vea el programa en crudo.
- `ui_instructions`: reglas de prompt a juego con las herramientas que lleva el agente.
- `OpenUiBackend`: validación y especificación aportadas por quien tenga la librería.
"""

from nas.adk.ui.openui import (
    FORM_INTERRUPT,
    OPENUI_PRESENT,
    OPENUI_PROMPT,
    OPENUI_SPEC,
    PROFILES,
    SUGGEST_FOLLOWUPS,
    FollowupTools,
    OpenUiBackend,
    OpenUiBackendError,
    OpenUiTools,
    ui_tools,
    unwrap,
)
from nas.adk.ui.prompts import (
    FOLLOWUP_RULES,
    TOOL_CALL_RULE,
    UI_REQUESTED_RULES,
    UI_RULES,
    UI_SPEC_HEADER,
    UI_SPEC_HINT,
    ui_instructions,
)
from nas.adk.ui.textcalls import RECOVERABLE_TOOLS, TextCallRecovery

__all__ = [
    "FOLLOWUP_RULES",
    "FORM_INTERRUPT",
    "FollowupTools",
    "OPENUI_PRESENT",
    "OPENUI_PROMPT",
    "OPENUI_SPEC",
    "OpenUiBackend",
    "OpenUiBackendError",
    "OpenUiTools",
    "PROFILES",
    "RECOVERABLE_TOOLS",
    "SUGGEST_FOLLOWUPS",
    "TOOL_CALL_RULE",
    "TextCallRecovery",
    "UI_REQUESTED_RULES",
    "UI_RULES",
    "UI_SPEC_HEADER",
    "UI_SPEC_HINT",
    "ui_instructions",
    "ui_tools",
    "unwrap",
]
