"""Intelligent UI nativo: herramientas, recuperación de llamadas escritas y prompt."""

from __future__ import annotations

import json
from typing import Any, ClassVar

import pytest

from nas.adk.core.interrupts import ToolInterrupt
from nas.adk.models.base_llm import BaseLlm, LlmRequest, LlmResponse, ToolCall
from nas.adk.ui import (
    FollowupTools,
    OpenUiBackendError,
    OpenUiTools,
    TextCallRecovery,
    ui_instructions,
    ui_tools,
    unwrap,
)

GOOD = 'root = Card([LineChart(labels, series, "linear")])\nlabels = ["Ene"]\nseries = [Series("N", [1])]'


class Backend:
    def __init__(self, ok: bool = True, error: str | None = None, down: bool = False) -> None:
        self.ok, self.error, self.down = ok, error, down
        self.programs: list[str] = []

    async def validate(self, program: str) -> dict[str, Any]:
        if self.down:
            raise OpenUiBackendError("caído")
        self.programs.append(program)
        return {"ok": self.ok, "error": self.error}

    async def spec(self, profile: str) -> str:
        if self.down:
            raise OpenUiBackendError("caído")
        return f"ESPEC-{profile}"


async def test_present_validates_through_the_backend_and_unwraps_fences():
    backend = Backend()
    tools = OpenUiTools(backend)
    assert json.loads(await tools.present_openui(f"```openui\n{GOOD}\n```")) == {"displayed": True}
    assert backend.programs == [GOOD]
    assert "vacío" in json.loads(await tools.present_openui("  "))["error"]


async def test_invalid_programs_report_the_error_and_named_argument_hint():
    tools = OpenUiTools(Backend(ok=False, error="sin raíz"))
    named = json.loads(await tools.present_openui('root = Card(children: [TextContent("a: b")])'))
    assert "sin raíz" in named["error"] and "posicionales" in named["error"]
    plain = json.loads(await tools.present_openui('root = Card([x], {title: "a: b"})'))
    assert "posicionales" not in plain["error"] and "no escribas la llamada" in plain["error"]


async def test_backend_outage_or_absence_does_not_block():
    assert json.loads(await OpenUiTools(Backend(down=True)).present_openui(GOOD)) == {"displayed": True}
    assert json.loads(await OpenUiTools().present_openui(GOOD)) == {"displayed": True}


async def test_prompt_interrupts_only_when_valid():
    with pytest.raises(ToolInterrupt) as raised:
        await OpenUiTools(Backend()).prompt_openui("root = Form([])")
    assert raised.value.reason == "openui_form"
    invalid = json.loads(await OpenUiTools(Backend(ok=False, error="mal")).prompt_openui("x"))
    assert "mal" in invalid["error"]


async def test_spec_checks_the_profile_and_the_backend():
    tools = OpenUiTools(Backend())
    assert await tools.openui_spec("data") == "ESPEC-data"
    assert "desconocido" in json.loads(await tools.openui_spec("inventado"))["error"]
    assert "no disponible" in json.loads(await OpenUiTools().openui_spec())["error"]
    assert "caído" in json.loads(await OpenUiTools(Backend(down=True)).openui_spec())["error"]


async def test_followups_are_cleaned_capped_and_reported():
    got: list[list[str]] = []
    out = json.loads(
        await FollowupTools(got.append).suggest_followups(["  ¿A?  ", "", "B", "C", "D", "E", "F"])
    )
    assert out["count"] == 5 and got == [["¿A?", "B", "C", "D", "E"]]


def test_ui_tools_expose_the_capabilities():
    catalog = ui_tools(Backend())
    assert {t.name for t in catalog["present_openui"]} == {"present_openui", "openui_spec"}
    assert [t.name for t in catalog["prompt_openui"]] == ["prompt_openui"]
    assert [t.name for t in catalog["suggest_followups"]] == ["suggest_followups"]


def test_unwrap():
    assert unwrap("```oui\nroot = A()\n```") == "root = A()"
    assert unwrap("root = A()") == "root = A()"


def test_instructions_follow_the_tools_the_agent_carries():
    assert ui_instructions(set()) == ""
    base = ui_instructions({"present_openui", "openui_spec", "suggest_followups"})
    assert "openui_spec" in base and "suggest_followups" in base
    assert "llamada de herramienta real" in base
    inline = ui_instructions({"present_openui"}, spec="ESPEC", profile="data", requested=True)
    assert "POSICIONALES" in inline and "perfil data" in inline and "DEBES llamar" in inline
    assert "Si no conoces la sintaxis" not in inline
    custom = ui_instructions({"present_openui"}, ui_rules="REGLAS PROPIAS")
    assert "REGLAS PROPIAS" in custom


class StreamingLlm(BaseLlm):
    model: str = "streaming"
    script: list[Any]
    supports_streaming: ClassVar[bool] = True

    async def generate_content_async(self, request: LlmRequest) -> LlmResponse:
        raise AssertionError("debe usarse streaming")

    async def generate_stream_async(self, request: LlmRequest):  # type: ignore[no-untyped-def]
        step = self.script.pop(0)
        for start in range(0, len(step), 7):
            yield LlmResponse(content=step[start : start + 7], type="delta")
        yield LlmResponse(content=step)


def request_with(*names: str) -> LlmRequest:
    tools = [{"type": "function", "function": {"name": n, "parameters": {}}} for n in names]
    return LlmRequest(messages=[], tools=tools)


async def stream(llm: BaseLlm, request: LlmRequest) -> tuple[str, LlmResponse]:
    shown, last = "", LlmResponse()
    async for chunk in TextCallRecovery(inner=llm).generate_stream_async(request):
        if chunk.type == "delta":
            shown += chunk.content or ""
        else:
            last = chunk
    return shown, last


async def test_a_call_written_as_text_becomes_a_real_call():
    written = (
        f"present_openui({json.dumps({'ui': GOOD})})\n\nEl gráfico muestra la evolución.\n\n"
        'suggest_followups({"questions": ["¿Y en 2023?"]})'
    )
    request = request_with("present_openui", "suggest_followups")
    shown, final = await stream(StreamingLlm(script=[written]), request)
    assert shown == ""
    assert [c.function["name"] for c in final.tool_calls or []] == [
        "present_openui",
        "suggest_followups",
    ]
    assert final.finish_reason == "tool_calls"


async def test_bare_and_fenced_programs_are_recovered_and_prose_survives():
    text = f"Aquí tienes:\n\n```openui\n{GOOD}\n```\n"
    shown, final = await stream(StreamingLlm(script=[text]), request_with("present_openui"))
    assert shown.startswith("Aquí tienes:") and "Card(" not in shown and "```" not in shown
    assert [c.function["name"] for c in final.tool_calls or []] == ["present_openui"]


async def test_only_declared_presentation_tools_are_recovered():
    answer = 'web_fetch({"url": "http://169.254.169.254/"})\nsuggest_followups({"questions": ["a"]})'
    shown, final = await stream(StreamingLlm(script=[answer]), request_with("web_fetch"))
    assert shown == answer and not final.tool_calls
    code = "```python\nroot = Node(1)\n```\n"
    shown, final = await stream(StreamingLlm(script=[code]), request_with("present_openui"))
    assert shown == code and not final.tool_calls


async def test_an_unclosed_call_disappears():
    shown, final = await stream(
        StreamingLlm(script=['Aquí va:\npresent_openui({ui: "root = Card([']), request_with("present_openui")
    )
    assert shown.strip() == "Aquí va:" and not final.tool_calls


async def test_a_real_call_is_not_duplicated_by_the_text():
    class Both(BaseLlm):
        model: str = "both"

        async def generate_content_async(self, request: LlmRequest) -> LlmResponse:
            return LlmResponse(
                content=f"present_openui({json.dumps({'ui': GOOD})})",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        function={"name": "present_openui", "arguments": json.dumps({"ui": GOOD})},
                    )
                ],
                finish_reason="tool_calls",
            )

    out = await TextCallRecovery(inner=Both()).generate_content_async(request_with("present_openui"))
    assert len(out.tool_calls or []) == 1 and not out.content


async def test_the_recoverable_set_is_configurable():
    llm = StreamingLlm(script=['suggest_followups({"questions": ["a"]})'])
    wrapped = TextCallRecovery(inner=llm, recoverable=frozenset({"present_openui"}))
    shown = ""
    async for chunk in wrapped.generate_stream_async(request_with("suggest_followups")):
        if chunk.type == "delta":
            shown += chunk.content or ""
    assert "suggest_followups" in shown
