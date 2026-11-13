"""Intelligent UI nativo: el modelo entrega interfaces como llamadas a herramienta.

Independiente de la aplicación que lo use. El lenguaje es OpenUI Lang; como su librería
de componentes es de JavaScript, la validación y la especificación las aporta un
`OpenUiBackend` (p. ej. un servicio que ejecuta el parser real). Sin backend las
herramientas siguen funcionando, solo que sin validar.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from nas.adk.core.interrupts import ToolInterrupt
from nas.adk.tools.base_tool import BaseTool
from nas.adk.tools.function_tool import FunctionTool

PROFILES = ("minimal", "data", "forms", "lists", "full")
MAX_FOLLOWUPS = 5
FORM_INTERRUPT = "openui_form"

OPENUI_PRESENT = "present_openui"
OPENUI_PROMPT = "prompt_openui"
OPENUI_SPEC = "openui_spec"
SUGGEST_FOLLOWUPS = "suggest_followups"


class OpenUiBackendError(RuntimeError):
    """El backend no está disponible; las herramientas no bloquean por ello."""


@runtime_checkable
class OpenUiBackend(Protocol):
    """Validación y especificación del lenguaje, a cargo de quien tenga la librería."""

    async def validate(self, program: str) -> dict[str, Any]:
        """`{"ok": bool, "error": str | None}`. Lanza `OpenUiBackendError` si no responde."""
        ...

    async def spec(self, profile: str) -> str:
        """Especificación de componentes del perfil. Lanza `OpenUiBackendError` si no responde."""
        ...


_FENCE = re.compile(r"^\s*```[ \t]*(?:openui|oui)?[ \t]*\n(.*?)(?:\n[ \t]*```)?\s*$", re.DOTALL | re.IGNORECASE)


_NAMED_ARGUMENT = re.compile(r"[A-Za-z_]\w*\s*(?::|=(?!=))")
_NAMED_ARGUMENT_HINT = (
    'Los argumentos de los componentes son posicionales: escribe Card([a, b]) y '
    'LineChart(labels, series, "linear", "Mes", "Precio"), nunca Card(children: [a, b]), '
    'variant: "linear" ni title="Hola".'
)
_RETRY_HINT = "Corrígelo y vuelve a llamar a la herramienta; no escribas la llamada en el texto."


def _has_named_arguments(ui: str) -> bool:
    """Detecta `Comp(nombre: valor)`; las claves de los objetos `{...}` sí son válidas."""
    stack: list[str] = []
    quote = False
    index = 0
    while index < len(ui):
        char = ui[index]
        if quote:
            if char == "\\":
                index += 1
            elif char == '"':
                quote = False
        elif char == '"':
            quote = True
        elif char in "([{":
            stack.append(char)
        elif char in ")]}":
            if stack:
                stack.pop()
        elif stack and stack[-1] == "(" and _NAMED_ARGUMENT.match(ui, index):
            return True
        index += 1
    return False


def _describe(ui: str, error: str) -> str:
    hint = f" {_NAMED_ARGUMENT_HINT}" if _has_named_arguments(ui) else ""
    return f"{error}.{hint} {_RETRY_HINT}"


def unwrap(ui: str) -> str:
    """Quita el bloque cercado con el que algunos modelos envuelven el programa."""
    match = _FENCE.match(ui or "")
    return (match.group(1) if match else ui or "").strip()


class OpenUiTools:
    """`present_openui`, `prompt_openui` y `openui_spec`."""

    def __init__(self, backend: OpenUiBackend | None = None) -> None:
        self._backend = backend

    async def _validate(self, ui: str) -> str | None:
        """Devuelve el error de validación, o None si el programa es válido."""
        ui = unwrap(ui)
        if not ui:
            return "el programa está vacío"
        if self._backend is None:
            return None
        try:
            verdict = await self._backend.validate(ui)
        except OpenUiBackendError:
            return None  # sin validador no se bloquea: el cliente vuelve a validar al pintar
        if verdict.get("ok"):
            return None
        return _describe(ui, str(verdict.get("error") or "programa no válido"))

    async def present_openui(self, ui: str) -> str:
        """Muestra una interfaz (tarjetas, tablas, gráficos…) escrita en OpenUI Lang.

        Empieza por `root = ...` y define todo lo que usa. Si devuelve un error, corrígelo
        y vuelve a llamar una vez.
        """
        error = await self._validate(ui)
        if error:
            return json.dumps({"error": f"programa OpenUI no válido: {error}"})
        return json.dumps({"displayed": True})

    async def prompt_openui(self, ui: str) -> str:
        """Muestra un formulario o una elección y ESPERA a que la persona lo envíe.

        Usa esto cuando necesites datos del usuario antes de seguir (preferencias, fechas…).
        """
        error = await self._validate(ui)
        if error:
            return json.dumps({"error": f"programa OpenUI no válido: {error}"})
        raise ToolInterrupt(FORM_INTERRUPT, {})

    async def openui_spec(self, profile: str = "minimal") -> str:
        """Devuelve la especificación de componentes de OpenUI para un perfil.

        Perfiles: minimal (tarjetas y texto), data (tablas y gráficos), forms, lists, full.
        Llámala una vez antes de escribir una interfaz si no la conoces ya.
        """
        if profile not in PROFILES:
            return json.dumps({"error": f"perfil desconocido; usa uno de {list(PROFILES)}"})
        if self._backend is None:
            return json.dumps({"error": "especificación no disponible"})
        try:
            return await self._backend.spec(profile)
        except OpenUiBackendError as error:
            return json.dumps({"error": str(error)})


class FollowupTools:
    """`suggest_followups`: el destino de las preguntas lo decide la aplicación."""

    def __init__(self, on_followups: Callable[[list[str]], None] | None = None) -> None:
        self._on_followups = on_followups
        self.questions: list[str] = []

    async def suggest_followups(self, questions: list[str]) -> str:
        """Propone 3 a 5 preguntas de seguimiento cortas y útiles. Llámala al final."""
        cleaned = [q.strip() for q in questions if isinstance(q, str) and q.strip()]
        self.questions = cleaned[:MAX_FOLLOWUPS]
        if self._on_followups is not None:
            self._on_followups(self.questions)
        return json.dumps({"ok": True, "count": len(self.questions)})


def ui_tools(
    backend: OpenUiBackend | None = None,
    on_followups: Callable[[list[str]], None] | None = None,
) -> dict[str, list[BaseTool]]:
    """Herramientas de UI por nombre de capacidad (para que el agente elija cuáles lleva)."""
    ui = OpenUiTools(backend)
    follow = FollowupTools(on_followups)
    return {
        OPENUI_PRESENT: [FunctionTool(ui.present_openui), FunctionTool(ui.openui_spec)],
        OPENUI_PROMPT: [FunctionTool(ui.prompt_openui)],
        SUGGEST_FOLLOWUPS: [FunctionTool(follow.suggest_followups)],
    }
