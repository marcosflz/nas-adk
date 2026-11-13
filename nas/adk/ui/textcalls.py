"""Recupera las llamadas a herramienta que el modelo escribe como texto.

Algunos modelos, sobre todo tras un error de herramienta, escriben
``present_openui({ui: "..."})`` en la respuesta en lugar de hacer la llamada. El
Runner lo trataría como respuesta final: el usuario vería el programa en crudo y
nada se pintaría. Aquí esas líneas no llegan al usuario y se convierten en
llamadas de verdad, que pasan por la misma política y la misma validación.

Otros escriben el programa OpenUI a secas (``root = Card([...])``), sin llamada
alguna: ese bloque se trata igual, como una llamada a ``present_openui``.

Solo se recuperan las herramientas de presentación, sin efectos fuera de la
conversación: un texto que imite una llamada (p. ej. citado de una web) no debe
poder disparar nada con efectos.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import AsyncGenerator
from typing import Any, ClassVar

from nas.adk.models.base_llm import BaseLlm, LlmRequest, LlmResponse, ModelCapabilities, ToolCall

OPENUI_PRESENT = "present_openui"
#: Herramientas de presentación recuperables por defecto (sin efectos fuera de la conversación).
RECOVERABLE_TOOLS = frozenset({OPENUI_PRESENT, "prompt_openui", "suggest_followups"})

_OPENERS = "([{"
_CLOSERS = ")]}"
_BARE_KEY = re.compile(r"^\{\s*([A-Za-z_]\w*)\s*:")
_UNESCAPED_QUOTE = re.compile(r'(?<!\\)"')


def _parse_args(body: str) -> dict[str, Any] | None:
    """Interpreta los argumentos: JSON, admitiendo la clave sin comillas (todas tienen una)."""
    source = _BARE_KEY.sub(lambda m: '{"' + m.group(1) + '":', body.strip(), count=1)
    try:
        args = json.loads(source, strict=False)
    except ValueError:
        return None
    if not isinstance(args, dict):
        return None
    if isinstance(args.get("ui"), str):
        args["ui"] = _undo_double_escape(args["ui"])
    return args


def _undo_double_escape(program: str) -> str:
    """Al escribirlo como texto, el programa puede quedar escapado una vez de más."""
    if '"' not in program or _UNESCAPED_QUOTE.search(program):
        return program
    try:
        decoded = json.loads(f'"{program}"', strict=False)
    except ValueError:
        return program
    return decoded if isinstance(decoded, str) else program


_PROGRAM_START = re.compile(r"^[ \t]*root[ \t]*=[ \t]*[A-Z]\w*[ \t]*\(")
_PROGRAM_START_PREFIX = re.compile(r"root[ \t]*(=[ \t]*([A-Z]\w*[ \t]*)?)?")
_STATEMENT = re.compile(r"^[ \t]*[A-Za-z_$][\w$]*[ \t]*=(?!=)")
_FENCE_LINE = re.compile(r"^[ \t]*```[ \t]*([\w+-]*)[ \t]*$")
_OPENUI_FENCES = frozenset({"", "openui", "oui"})


class _ProgramFilter:
    """Convierte un programa OpenUI escrito a secas en una llamada escrita como texto.

    El programa es un bloque de sentencias `nombre = Expresión` que empieza por
    `root = Componente(`, con o sin bloque cercado. Sale como
    `present_openui({"ui": ...})`, que `_TextCallFilter` ya sabe recuperar. El código
    de un bloque cercado de otro lenguaje (```python) no se toca.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._mid_line = False
        self._code_fence = False
        self._held_fence: str | None = None
        self._program: list[str] | None = None
        self._depth = 0
        self._quote = False
        self._escaped = False

    def feed(self, text: str) -> str:
        self._buffer += text
        out: list[str] = []
        while (newline := self._buffer.find("\n")) >= 0:
            line, self._buffer = self._buffer[: newline + 1], self._buffer[newline + 1 :]
            if self._mid_line:
                out.append(line)
                self._mid_line = False
            else:
                self._line(line, out)
        if self._buffer and (self._mid_line or not self._may_matter(self._buffer)):
            out.append(self._buffer)
            self._buffer = ""
            self._mid_line = True
        return "".join(out)

    def flush(self) -> str:
        out: list[str] = []
        if self._buffer:
            if self._mid_line:
                out.append(self._buffer)
            else:
                self._line(self._buffer, out)
            self._buffer = ""
        if self._program is not None:
            self._end_program(out)
        if self._held_fence is not None:
            out.append(self._held_fence)
            self._held_fence = None
        return "".join(out)

    def _may_matter(self, partial: str) -> bool:
        """Una línea a medias se retiene mientras pueda abrir o continuar un programa."""
        if self._program is not None or self._held_fence is not None:
            return True
        if self._code_fence:
            return partial.lstrip(" \t").startswith("`")
        head = partial.lstrip(" \t")
        if not head or "```".startswith(head) or _FENCE_LINE.match(partial):
            return True
        # Un programa de una sola línea llega entero antes de su salto de línea.
        return "root".startswith(head) or bool(_PROGRAM_START_PREFIX.fullmatch(head) or _PROGRAM_START.match(partial))

    def _line(self, line: str, out: list[str]) -> None:
        fence = _FENCE_LINE.match(line.rstrip("\r\n"))
        if self._program is not None:
            if self._depth > 0 or self._quote:
                self._take(line)
                return
            if fence:  # cierre del bloque cercado que envolvía el programa
                self._end_program(out)
                return
            if not line.strip() or _STATEMENT.match(line):
                self._take(line)
                return
            self._end_program(out)
        if self._code_fence:
            out.append(line)
            self._code_fence = not fence
            return
        if self._held_fence is not None:
            if not line.strip():
                self._held_fence += line
                return
            held, self._held_fence = self._held_fence, None
            if _PROGRAM_START.match(line):
                self._start_program(line)
                return
            out.append(held)
            out.append(line)
            self._code_fence = not fence
            return
        if fence:
            if fence.group(1).lower() in _OPENUI_FENCES:
                self._held_fence = line
            else:
                out.append(line)
                self._code_fence = True
            return
        if _PROGRAM_START.match(line):
            self._start_program(line)
            return
        out.append(line)

    def _start_program(self, line: str) -> None:
        self._program = []
        self._depth, self._quote, self._escaped = 0, False, False
        self._take(line)

    def _take(self, line: str) -> None:
        assert self._program is not None
        self._program.append(line)
        for char in line:
            if self._quote:
                if self._escaped:
                    self._escaped = False
                elif char == "\\":
                    self._escaped = True
                elif char == '"':
                    self._quote = False
            elif char == '"':
                self._quote = True
            elif char in _OPENERS:
                self._depth += 1
            elif char in _CLOSERS:
                self._depth -= 1

    def _end_program(self, out: list[str]) -> None:
        program = "".join(self._program or []).strip()
        complete = self._depth == 0 and not self._quote
        self._program = None
        if program and complete:  # cortado a medias: mejor nada que el programa en crudo
            out.append(f"{OPENUI_PRESENT}({json.dumps({'ui': program}, ensure_ascii=False)})\n")


class _TextCallFilter:
    """Separa, también en streaming, la prosa de las llamadas escritas como texto.

    Solo cuenta como llamada la que empieza una línea, para no tocar la prosa que
    mencione una herramienta. Lo que el modelo escribe después de una llamada
    recuperada se descarta: es lo que imagina que devolvió una herramienta que aún
    no se ha ejecutado, y responderá de verdad en el turno siguiente.
    """

    def __init__(self, names: frozenset[str]) -> None:
        self._names = names
        alternatives = "|".join(re.escape(name) for name in sorted(names))
        self._pattern = re.compile(rf"^[ \t]*({alternatives})[ \t]*\(", re.MULTILINE)
        self._pending = ""
        self._at_line_start = True
        self._muted = False
        self._started = False
        self._reset_scan()
        self._call: str | None = None
        self._programs = _ProgramFilter() if OPENUI_PRESENT in names else None
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _reset_scan(self) -> None:
        self._position = 0
        self._depth = 1
        self._quote = ""
        self._escaped = False

    def feed(self, text: str) -> str:
        self._pending += self._programs.feed(text) if self._programs else text
        return self._drain(final=False)

    def flush(self) -> str:
        if self._programs:
            self._pending += self._programs.flush()
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> str:
        out: list[str] = []

        def emit(text: str) -> None:
            if text and not self._muted:
                out.append(text)

        while self._pending:
            if self._call is not None:
                end = self._closing_paren()
                if end is None:
                    if final:  # llamada cortada: mejor nada que el programa a medias
                        self._pending, self._call = "", None
                    break
                args = _parse_args(self._pending[:end])
                if args is not None:
                    self.calls.append((self._call, args))
                    self._muted = True
                self._pending = self._pending[end + 1 :]
                self._call = None
                self._at_line_start = False
                continue

            if not self._at_line_start:
                newline = self._pending.find("\n")
                if newline < 0:
                    emit(self._pending)
                    self._pending = ""
                    break
                emit(self._pending[: newline + 1])
                self._pending = self._pending[newline + 1 :]
                self._at_line_start = True
                continue

            match = self._pattern.search(self._pending)
            if match is not None:
                emit(self._pending[: match.start()])
                self._call = match.group(1)
                self._pending = self._pending[match.end() :]
                self._reset_scan()
                continue

            keep = 0 if final else self._undecided_tail()
            cut = len(self._pending) - keep
            emit(self._pending[:cut])
            if cut:
                self._at_line_start = keep > 0 or self._pending[cut - 1] == "\n"
            self._pending = self._pending[cut:]
            break

        text = "".join(out)
        if not self._started:  # sin burbuja vacía si la llamada era todo el mensaje
            text = text.lstrip()
            self._started = bool(text)
        return text

    def _undecided_tail(self) -> int:
        """Longitud de la última línea si aún podría acabar siendo una llamada."""
        tail = self._pending[self._pending.rfind("\n") + 1 :]
        head = tail.lstrip(" \t")
        if not head:
            return len(tail)
        for name in self._names:
            if name.startswith(head) or (head.startswith(name) and not head[len(name) :].strip(" \t")):
                return len(tail)
        return 0

    def _closing_paren(self) -> int | None:
        """Índice del paréntesis que cierra la llamada; sigue donde lo dejó el trozo anterior."""
        text = self._pending
        while self._position < len(text):
            char = text[self._position]
            if self._quote:
                if self._escaped:
                    self._escaped = False
                elif char == "\\":
                    self._escaped = True
                elif char == self._quote:
                    self._quote = ""
            elif char == '"':
                self._quote = char
            elif char in _OPENERS:
                self._depth += 1
            elif char in _CLOSERS:
                self._depth -= 1
                if self._depth == 0:
                    return self._position
            self._position += 1
        return None


class TextCallRecovery(BaseLlm):
    """Envuelve un modelo para que no se filtren llamadas escritas como texto."""

    inner: BaseLlm
    #: Herramientas que se recuperan si el agente las declara; deben ser solo de presentación.
    recoverable: frozenset[str] = RECOVERABLE_TOOLS
    #: Pide al proveedor el resumen del razonamiento (si lo soporta) en cada petición.
    reasoning_summary: bool = False
    # Los modelos sin streaming entregan su única respuesta por el método por defecto.
    supports_streaming: ClassVar[bool] = True

    def model_post_init(self, _context: Any) -> None:
        self.model = self.inner.model

    def get_capabilities(self) -> ModelCapabilities | None:
        return self.inner.get_capabilities()

    async def aclose(self) -> None:
        await self.inner.aclose()

    async def decide_async(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        return await self.inner.decide_async(state, questions)

    def _prepare(self, request: LlmRequest) -> LlmRequest:
        if self.reasoning_summary:
            return request.model_copy(update={"reasoning_summary": True})
        return request

    async def generate_content_async(self, request: LlmRequest) -> LlmResponse:
        request = self._prepare(request)
        return _recover(
            await self.inner.generate_content_async(request), _recoverable(request, self.recoverable)
        )

    async def generate_stream_async(
        self, request: LlmRequest
    ) -> AsyncGenerator[LlmResponse, None]:
        names = _recoverable(request, self.recoverable)
        live = _TextCallFilter(names) if names else None
        request = self._prepare(request)
        async for chunk in self.inner.generate_stream_async(request):
            if chunk.type == "delta" and chunk.reasoning:
                yield chunk
                continue
            if chunk.type != "delta":
                if live is not None and (rest := live.flush()):
                    yield LlmResponse(content=rest, type="delta")
                yield _recover(chunk, names)
                continue
            if live is None:
                yield chunk
            elif text := live.feed(chunk.content or ""):
                yield LlmResponse(content=text, type="delta")


def _recoverable(request: LlmRequest, allowed: frozenset[str]) -> frozenset[str]:
    declared = {(tool.get("function") or {}).get("name") for tool in request.tools or []}
    return allowed & declared


def _recover(response: LlmResponse, names: frozenset[str]) -> LlmResponse:
    """Limpia la respuesta consolidada y pasa a ``tool_calls`` lo que estaba en el texto."""
    if not names or not response.content:
        return response
    calls = _TextCallFilter(names)
    text = calls.feed(response.content) + calls.flush()
    if not calls.calls and text.strip() == response.content.strip():
        return response
    update: dict[str, Any] = {"content": text.strip() or None}
    if calls.calls and not response.tool_calls and response.finish_reason in {"stop", "completed"}:
        update["tool_calls"] = [
            ToolCall(
                id=f"text-call-{uuid.uuid4().hex}",
                function={"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            )
            for name, args in calls.calls
        ]
        update["finish_reason"] = "tool_calls"
    return response.model_copy(update=update)
