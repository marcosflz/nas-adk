from __future__ import annotations

import json
from typing import Any


from nas.adk import Agent, Runner
from nas.adk.core.callbacks import BaseCallback
from nas.adk.core.interrupts import ResumeEntry, ToolDecision, ToolInterrupt
from nas.adk.models.base_llm import BaseLlm, LlmRequest, LlmResponse, ToolCall
from nas.adk.sessions.in_memory_session_service import InMemorySessionService
from nas.adk.tools.function_tool import FunctionTool


class ScriptedLlm(BaseLlm):
    """Devuelve las respuestas del guion en orden y recuerda lo que recibió."""

    model: str = "scripted"
    script: list[LlmResponse]
    seen: list[list[dict[str, Any]]] = []

    async def generate_content_async(self, request: LlmRequest) -> LlmResponse:
        self.seen.append(list(request.messages))
        return self.script.pop(0)


def call(call_id: str, name: str, **args: Any) -> ToolCall:
    return ToolCall(id=call_id, function={"name": name, "arguments": json.dumps(args)})


def asks(*calls: ToolCall, **usage: int) -> LlmResponse:
    return LlmResponse(tool_calls=list(calls), finish_reason="tool_calls", usage=usage)


def says(text: str, **usage: int) -> LlmResponse:
    return LlmResponse(content=text, usage=usage)


async def collect(runner: Runner, **kwargs: Any) -> list[Any]:
    return [e async for e in runner.run_async("u", "s", **kwargs)]


def build(script: list[LlmResponse], tools: list[Any], **runner_kwargs: Any):
    llm = ScriptedLlm(script=script, seen=[])
    agent = Agent(name="a", model=llm, instruction="x", tools=tools)
    return llm, Runner(agent=agent, session_service=InMemorySessionService(), **runner_kwargs)


def ask_user(question: str) -> str:
    """Pregunta algo a la persona."""
    raise ToolInterrupt("form", {"question": question})


def send_email(to: str) -> str:
    """Envía un correo."""
    return f"enviado a {to}"


class AskBeforeEmail(BaseCallback):
    async def before_tool_call(self, tool_name, args, tool_call_id="", **_):
        if tool_name == "send_email":
            return ToolDecision.interrupt("approval", {"to": args["to"]})
        return None


async def test_a_tool_can_interrupt_and_the_run_ends_as_interrupted() -> None:
    _, runner = build([asks(call("c1", "ask_user", question="¿color?"))], [ask_user])

    events = await collect(runner, message="hola")

    last = events[-1]
    assert (last.type, last.status, last.final) == ("status", "interrupted", True)
    interrupt = last.metrics["interrupts"][0]
    assert interrupt["id"] == "c1" and interrupt["toolName"] == "ask_user"
    assert interrupt["reason"] == "form" and interrupt["payload"] == {"question": "¿color?"}
    # La llamada queda sin resultado: no hay evento de tool.
    assert not [e for e in events if e.role == "tool"]


async def test_resume_with_a_human_answer_continues_the_loop() -> None:
    llm, runner = build(
        [asks(call("c1", "ask_user", question="¿color?")), says("Elegiste azul")], [ask_user]
    )
    await collect(runner, message="hola")

    events = await collect(
        runner, resume=[ResumeEntry("c1", "resolved", {"color": "azul"})]
    )

    tool_event = next(e for e in events if e.role == "tool")
    assert json.loads(tool_event.content) == {"color": "azul"}
    assert events[-1].status == "completed" and events[-1].content == "Elegiste azul"
    # Sin mensaje de usuario nuevo: el modelo ve la llamada y su resultado.
    assert not [e for e in events if e.role == "user"]
    roles = [m["role"] for m in llm.seen[-1]]
    assert roles[-2:] == ["assistant", "tool"]


async def test_policy_interrupt_then_approval_executes_without_asking_again() -> None:
    llm, runner = build(
        [asks(call("c1", "send_email", to="a@b.c")), says("hecho")],
        [send_email],
        callbacks=[AskBeforeEmail()],
    )

    first = await collect(runner, message="manda")
    assert first[-1].status == "interrupted"
    assert first[-1].metrics["interrupts"][0]["payload"] == {"to": "a@b.c"}
    assert not [e for e in first if e.role == "tool"]  # no se ejecutó

    second = await collect(runner, resume=[ResumeEntry("c1", "approved")])

    tool_event = next(e for e in second if e.role == "tool")
    assert tool_event.content == "enviado a a@b.c"
    assert second[-1].status == "completed"


async def test_denied_resume_gives_the_model_an_error_instead_of_running() -> None:
    _, runner = build(
        [asks(call("c1", "send_email", to="a@b.c")), says("no se envió")],
        [send_email],
        callbacks=[AskBeforeEmail()],
    )
    await collect(runner, message="manda")

    events = await collect(runner, resume=[ResumeEntry("c1", "denied", reason="no")])

    tool_event = next(e for e in events if e.role == "tool")
    assert json.loads(tool_event.content) == {"error": "no"}
    assert "enviado" not in tool_event.content


async def test_a_deny_decision_returns_an_error_without_running_the_tool() -> None:
    class DenyAll(BaseCallback):
        async def before_tool_call(self, tool_name, args, **_):
            return ToolDecision.deny("bloqueado por política")

    _, runner = build(
        [asks(call("c1", "send_email", to="x")), says("vale")],
        [send_email],
        callbacks=[DenyAll()],
    )

    events = await collect(runner, message="manda")

    tool_event = next(e for e in events if e.role == "tool")
    assert json.loads(tool_event.content) == {"error": "bloqueado por política"}
    assert events[-1].status == "completed"


async def test_remaining_calls_are_picked_up_on_resume() -> None:
    # Dos llamadas: la primera se interrumpe; la segunda no llega a ejecutarse
    # hasta la reanudación y entonces se ejecuta con normalidad.
    llm, runner = build(
        [
            asks(call("c1", "ask_user", question="?"), call("c2", "send_email", to="z")),
            says("fin"),
        ],
        [ask_user, send_email],
    )
    first = await collect(runner, message="hola")
    assert first[-1].status == "interrupted"

    events = await collect(runner, resume=[ResumeEntry("c1", "resolved", "sí")])

    results = {e.tool_call_id: e.content for e in events if e.role == "tool"}
    assert results == {"c1": "sí", "c2": "enviado a z"}


async def test_resume_validates_its_input() -> None:
    _, runner = build([says("hola")], [ask_user])
    await collect(runner, message="hola")

    nothing = await collect(runner, resume=[ResumeEntry("c9", "resolved", 1)])
    assert nothing[-1].status == "error"
    assert "no pending tool calls" in nothing[-1].content

    _, runner2 = build([asks(call("c1", "ask_user", question="?"))], [ask_user])
    await collect(runner2, message="hola")
    unknown = await collect(runner2, resume=[ResumeEntry("otra", "resolved", 1)])
    assert unknown[-1].status == "error" and "Unknown tool call ids" in unknown[-1].content

    both = [e async for e in runner2.run_async("u", "s", message="x", resume=[ResumeEntry("c1")])]
    assert both[-1].status == "error"


async def test_interrupted_session_is_not_treated_as_failed_history() -> None:
    llm, runner = build(
        [asks(call("c1", "ask_user", question="?")), says("ok"), says("otra")], [ask_user]
    )
    await collect(runner, message="primero")
    await collect(runner, resume=[ResumeEntry("c1", "resolved", "r")])

    await collect(runner, message="segundo")

    texts = [m.get("content") for m in llm.seen[-1]]
    assert "primero" in texts and "segundo" in texts  # el turno interrumpido se conserva


async def test_token_budget_stops_the_run() -> None:
    _, runner = build(
        [
            asks(call("c1", "send_email", to="a"), total_tokens=900),
            asks(call("c2", "send_email", to="b"), total_tokens=900),
            says("nunca"),
        ],
        [send_email],
        max_total_tokens=1000,
    )

    events = await collect(runner, message="hola")

    assert events[-1].status == "budget_exhausted"
    assert "Token budget" in events[-1].content


async def test_legacy_callbacks_without_the_new_arguments_still_work() -> None:
    seen: list[str] = []

    class Legacy(BaseCallback):
        async def before_tool_call(self, tool_name: str, args: dict[str, Any]) -> None:
            seen.append(tool_name)

    _, runner = build(
        [asks(call("c1", "send_email", to="a")), says("ok")], [send_email], callbacks=[Legacy()]
    )
    events = await collect(runner, message="hola")

    assert seen == ["send_email"] and events[-1].status == "completed"


async def test_callbacks_receive_the_call_id_and_context() -> None:
    got: dict[str, Any] = {}

    class Spy(BaseCallback):
        async def before_tool_call(self, tool_name, args, tool_call_id="", ctx=None, agent_name=""):
            got.update(id=tool_call_id, ctx=ctx, agent=agent_name)

    _, runner = build(
        [asks(call("c7", "send_email", to="a")), says("ok")], [send_email], callbacks=[Spy()]
    )
    await collect(runner, message="hola")

    assert got["id"] == "c7" and got["agent"] == "a" and got["ctx"] is not None


def test_function_tool_exceptions_still_become_errors() -> None:
    # Una excepción normal NO interrumpe: solo ToolInterrupt lo hace.
    def boom() -> str:
        """Falla."""
        raise RuntimeError("fallo")

    assert FunctionTool(boom).name == "boom"
