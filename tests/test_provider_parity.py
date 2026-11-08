"""La misma función Python, con los mismos permisos, funciona con ambos proveedores."""

from __future__ import annotations

import json
from typing import Any

import pytest

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")

from nas.adk import Agent, Runner  # noqa: E402
from nas.adk.core.callbacks import BaseCallback  # noqa: E402
from nas.adk.core.interrupts import ToolDecision  # noqa: E402
from nas.adk.models.anthropic import AnthropicLlm  # noqa: E402
from nas.adk.models.claude_code import ClaudeCodeLlm  # noqa: E402
from nas.adk.sessions.sqlite_session_service import SQLiteSessionService  # noqa: E402
from tests.claude_code_fakes import INIT, FakeHandle, FakeTransport, calls, final  # noqa: E402

executions: list[str] = []


def consultar_gastos(mes: str) -> str:
    """Devuelve el total de gastos de un mes (YYYY-MM)."""
    executions.append(mes)
    return json.dumps({"mes": mes, "total": 42})


class DenyGastos(BaseCallback):
    async def before_tool_call(self, tool_name: str, args: dict[str, Any], **_: Any) -> ToolDecision | None:
        return ToolDecision.deny("sin permiso")


def anthropic_msg(content: list[dict[str, Any]], stop: str) -> dict[str, Any]:
    return {
        "id": "m", "type": "message", "role": "assistant", "model": "c", "content": content,
        "stop_reason": stop, "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def make_anthropic(responses: list[dict[str, Any]], bodies: list[dict[str, Any]]) -> tuple[AnthropicLlm, Any]:
    def handler(request: Any) -> Any:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=responses.pop(0))

    http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    client = anthropic.AsyncAnthropic(api_key="k", http_client=http, max_retries=0)
    return AnthropicLlm(model="c", client=client), http


TOOL_USE = [{"type": "tool_use", "id": "toolu_1", "name": "consultar_gastos", "input": {"mes": "2026-11"}}]


def anthropic_script(*, final_text: str = "Gastaste 42") -> list[dict[str, Any]]:
    return [
        anthropic_msg(TOOL_USE, "tool_use"),
        anthropic_msg([{"type": "text", "text": final_text}], "end_turn"),
    ]


def claude_code_llm(*runs: dict[str, Any]) -> tuple[ClaudeCodeLlm, FakeTransport]:
    t = FakeTransport([FakeHandle([INIT, r]) for r in runs])
    return ClaudeCodeLlm(transport=t), t


async def run(model: Any, *, callbacks: list[Any] | None = None, **kw: Any) -> list[Any]:
    agent = Agent(name="a", model=model, instruction="Eres un asistente.", tools=[consultar_gastos])
    runner = Runner(agent=agent, callbacks=callbacks, **kw)
    return [e async for e in runner.run_async("u", "s", "¿cuánto gasté en noviembre?")]


@pytest.fixture(autouse=True)
def reset() -> None:
    executions.clear()


@pytest.fixture(params=["anthropic", "claude-code"])
def provider(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def build(provider: str, *, steps: int = 1, final_text: str = "Gastaste 42") -> tuple[Any, list[Any], Any]:
    """Modelo que pide la herramienta ``steps`` veces y luego responde."""
    if provider == "anthropic":
        script = [anthropic_msg(TOOL_USE, "tool_use") for _ in range(steps)]
        script.append(anthropic_msg([{"type": "text", "text": final_text}], "end_turn"))
        bodies: list[dict[str, Any]] = []
        model, http = make_anthropic(script, bodies)
        return model, bodies, http
    runs = [calls(("consultar_gastos", {"mes": "2026-11"})) for _ in range(steps)] + [final(final_text)]
    model, transport = claude_code_llm(*runs)
    return model, [], transport


async def test_same_python_function_works_with_both_providers(provider: str) -> None:
    model, _, extra = build(provider)
    events = await run(model)
    assert events[-1].content == "Gastaste 42"
    assert executions == ["2026-11"]  # ejecutada una vez, por el Runner
    tool_events = [e for e in events if e.role == "tool"]
    assert json.loads(tool_events[0].content or "")["total"] == 42
    asked = next(e for e in events if e.tool_calls)
    assert asked.tool_calls[0]["id"] == tool_events[0].tool_call_id  # IDs enlazados
    if provider == "anthropic":
        await extra.aclose()


async def test_identical_permissions_deny_without_executing(provider: str) -> None:
    model, bodies, extra = build(provider, final_text="No puedo")
    events = await run(model, callbacks=[DenyGastos()])
    assert executions == []  # el proveedor no puede autoaprobar
    tool = next(e for e in events if e.role == "tool")
    assert json.loads(tool.content or "") == {"error": "sin permiso"}
    assert events[-1].content == "No puedo"
    if provider == "anthropic":
        results = bodies[1]["messages"][-1]["content"]
        assert results[0]["is_error"] is True
        await extra.aclose()


async def test_turn_limit_stops_the_loop(provider: str) -> None:
    model, _, extra = build(provider, steps=5)
    events = await run(model, max_turns=2)
    assert events[-1].status in {"error", "budget_exhausted"}
    assert len(executions) == 2
    if provider == "anthropic":
        await extra.aclose()


async def test_claude_code_invalid_output_never_executes_tools() -> None:
    model, _ = claude_code_llm(
        calls(("herramienta_inventada", {"x": 1})),
        calls(("consultar_gastos", {"mes": 5})),  # argumentos inválidos
    )
    model.protocol_repairs = 1
    events = await run(model)
    assert executions == []
    assert events[-1].status == "error"


async def test_claude_code_repair_then_single_execution() -> None:
    model, t = claude_code_llm(
        calls(("nope", {})),
        calls(("consultar_gastos", {"mes": "2026-11"})),
        final("ok"),
    )
    events = await run(model)
    assert events[-1].content == "ok"
    assert executions == ["2026-11"]  # el reintento del modelo no repite herramientas


async def test_sqlite_roundtrip_keeps_anthropic_native_blocks(tmp_path: Any) -> None:
    thinking = {"type": "thinking", "thinking": "plan", "signature": "sig"}
    script = [
        anthropic_msg([thinking, *TOOL_USE], "tool_use"),
        anthropic_msg([{"type": "text", "text": "uno"}], "end_turn"),
        anthropic_msg([{"type": "text", "text": "dos"}], "end_turn"),
    ]
    bodies: list[dict[str, Any]] = []
    model, http = make_anthropic(script, bodies)
    db = tmp_path / "s.db"
    agent = Agent(name="a", model=model, instruction="x", tools=[consultar_gastos])

    runner = Runner(agent=agent, session_service=SQLiteSessionService(str(db)))
    first = [e async for e in runner.run_async("u", "s", "primero")]
    assert first[-1].content == "uno"

    # Sesión nueva desde disco: el segundo turno reenvía el bloque thinking firmado.
    runner2 = Runner(agent=agent, session_service=SQLiteSessionService(str(db)))
    second = [e async for e in runner2.run_async("u", "s", "segundo")]
    assert second[-1].content == "dos"
    replayed = bodies[2]["messages"]
    assert replayed[1]["content"][0] == thinking
    assert executions == ["2026-11"]
    await http.aclose()

