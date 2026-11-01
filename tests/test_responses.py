from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel
from rich.console import Console

from nas.adk import Agent, Runner
from nas.adk.models import ModelRouter
from nas.adk.models.base_llm import LlmRequest
from nas.adk.models.router import _build_request_body, _build_responses_body
from nas.adk.sessions import InMemorySessionService, SQLiteSessionService


def response(number: int, *, tool: bool = False) -> dict:
    output = [{"type": "reasoning", "id": f"rs-{number}", "summary": [], "encrypted_content": f"encrypted-{number}"}]
    output.append({
        "type": "message", "id": f"msg-{number}", "role": "assistant", "status": "completed",
        "phase": "commentary" if tool else "final_answer",
        "content": [{"type": "output_text", "text": "Consultando" if tool else '{"value":14}', "annotations": []}],
    })
    if tool:
        output.append({"type": "function_call", "id": "fc-item", "call_id": "call-1", "name": "lookup", "arguments": '{"value":7}', "status": "completed"})
    return {
        "id": f"resp-{number}", "object": "response", "status": "completed", "output": output,
        "usage": {"input_tokens": 12, "output_tokens": 5, "total_tokens": 17, "output_tokens_details": {"reasoning_tokens": 3}},
    }


def sse(data: dict) -> str:
    events = []
    for index, item in enumerate(data["output"]):
        if item["type"] == "message":
            for text in ("", item["content"][0]["text"]):
                events.append({"type": "response.output_text.delta", "delta": text, "item_id": item["id"], "output_index": index, "content_index": 0, "sequence_number": len(events)})
    events.append({"type": "response.completed", "response": data, "sequence_number": len(events)})
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("sqlite", [False, True])
async def test_responses_tools_reasoning_history_and_session_isolation(
    tmp_path: Path, stream: bool, sqlite: bool,
) -> None:
    requests, calls = [], []

    def lookup(value: int = 1) -> dict:
        calls.append(value)
        return {"value": value * 2}

    class Result(BaseModel):
        value: int

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/openai/v1/responses")
        assert request.headers["Authorization"] == "Bearer offline"
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) > 1:
            assert all("status" not in item for item in body["input"])
        assert body["reasoning"] == {"effort": "high"}
        assert body["store"] is False and body["include"] == ["reasoning.encrypted_content"]
        assert body["max_output_tokens"] == 50 and "max_tokens" not in body
        assert "messages" not in body and "reasoning_effort" not in body
        assert body["tools"][0]["name"] == "lookup" and body["tools"][0]["strict"] is False
        assert body["text"]["format"]["type"] == "json_schema" and "schema" in body["text"]["format"]
        data = response(len(requests), tool=len(requests) == 1)
        if body.get("stream"):
            return httpx.Response(200, text=sse(data), headers={"Content-Type": "text/event-stream"})
        return httpx.Response(200, json=data)

    path = str(tmp_path / "sessions.db")
    service = SQLiteSessionService(path) if sqlite else InMemorySessionService()
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            async with ModelRouter(
                provider="opencode",
                model="gpt-6-luna", api="responses",
                base_url="https://example.invalid/go",
                api_key="offline", http_client=client,
            ) as model:
                agent = Agent(name="test", model=model, tools=[lookup], output_schema=Result,
                              generation_config={"reasoning_effort": "high", "max_tokens": 50})
                runner = Runner(agent=agent, session_service=service)
                events = [event async for event in runner.run_async("user", "session", "consulta", stream=stream)]
                assert events[-1].status == "completed" and events[-1].content == '{"value":14}'
                assert all(event.content for event in events if event.type == "stream")
                assert events[-1].metrics["model_calls"] == 2 and calls == [7]
                assert events[-1].usage["completion_tokens_details"]["reasoning_tokens"] == 3
                assistant = next(event for event in events if event.tool_calls)
                assert assistant.tool_calls[0]["id"] == "call-1"
                assert assistant.response_items[1]["phase"] == "commentary"
                assert assistant.response_items[1]["status"] == "completed"
                replay = requests[1]["input"]
                replayed = [
                    item for item in replay
                    if item.get("type") in {"reasoning", "message", "function_call"}
                ]
                assert replayed == [
                    {key: value for key, value in item.items() if key != "status"}
                    for item in assistant.response_items
                ]
                tool_result = next(item for item in replay if item.get("type") == "function_call_output")
                assert tool_result["call_id"] == "call-1" and json.loads(tool_result["output"]) == {"value": 14}
                assert sum(item.get("type") == "function_call" for item in replay) == 1
                if sqlite:
                    service.close()
                    service = SQLiteSessionService(path)
                    runner = Runner(agent=agent, session_service=service)
                followup = [event async for event in runner.run_async("user", "session", "resume", stream=stream)]
                assert followup[-1].status == "completed"
                assert next(item for item in requests[2]["input"] if item.get("id") == "rs-1")["encrypted_content"] == "encrypted-1"
                assert any(item.get("id") == "rs-2" for item in requests[2]["input"])
                assert any(item.get("phase") == "final_answer" for item in requests[2]["input"])
                isolated = [event async for event in runner.run_async("other", "other-session", "consulta", stream=stream)]
                assert isolated[-1].status == "completed"
                assert not any(item.get("type") in {"reasoning", "function_call"} for item in requests[3]["input"])
    finally:
        if sqlite:
            service.close()


@pytest.mark.parametrize("ending", ["missing", "incomplete", "failed", "bad_call", "bad_role", "unsupported"])
async def test_responses_failure_never_executes_partial_tools(ending: str) -> None:
    calls = []

    def lookup(value: int) -> str:
        calls.append(value)
        return "must not run"

    data = response(1, tool=True)
    events = [{"type": "response.output_text.delta", "delta": "Parcial", "item_id": "msg-1", "output_index": 1, "content_index": 0, "sequence_number": 0}]
    if ending != "missing":
        status = ending if ending in {"incomplete", "failed"} else "completed"
        data["status"] = status
        data["incomplete_details"] = {"reason": "max_output_tokens"}
        if ending == "failed":
            data["error"] = {"code": "server_error", "message": "backend failed"}
        elif ending == "bad_call":
            data["output"][-1].pop("call_id")
        elif ending == "bad_role":
            data["output"][1]["role"] = "user"
        elif ending == "unsupported":
            data["output"][0]["type"] = "web_search_call"
        events.append({"type": f"response.{status}", "response": data, "sequence_number": 1})
    text = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, text=text, headers={"Content-Type": "text/event-stream"}),
    )) as client:
        async with ModelRouter(
            provider="opencode", model="gpt-6-luna", api="responses",
            base_url="https://example.invalid/go", api_key="offline", http_client=client,
        ) as model:
            output = [event async for event in Runner(agent=Agent(name="test", model=model, tools=[lookup])).run_async("user", "session", "Hola", stream=True)]
    assert not calls and output[-1].status == "error"
    assert not any(event.tool_calls for event in output)
    assert "Parcial" in "".join(event.content or "" for event in output if event.type == "stream")


def test_responses_legacy_history_and_private_items_never_reach_chat_api() -> None:
    messages = [
        {"role": "user", "content": "Hola"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call", "function": {"name": "lookup", "arguments": {"value": 1}}}]},
        {"role": "tool", "tool_call_id": "call", "content": "result"},
        {"role": "assistant", "content": "done", "_response_items": response(2)["output"]},
    ]
    request = LlmRequest(messages=messages, reasoning_effort="high")
    native = _build_responses_body(request)
    assert all("status" not in item for item in native["input"])
    assert native["input"][1]["arguments"] == '{"value": 1}'
    assert native["input"][2] == {"type": "function_call_output", "call_id": "call", "output": "result"}
    assert any(item.get("id") == "rs-2" for item in native["input"])
    chat = _build_request_body(request)
    assert all("_response_items" not in message for message in chat["messages"])
    assert "_response_items" in request.messages[-1]


@pytest.mark.parametrize("stream", [False, True])
async def test_responses_final_phase_validates_schema_without_commentary(stream) -> None:
    class Result(BaseModel):
        value: int

    data = response(1)
    preamble = {"type": "message", "id": "preamble", "role": "assistant", "status": "completed", "phase": "commentary",
                "content": [{"type": "output_text", "text": "Voy a comprobarlo.", "annotations": []}]}
    data["output"].insert(0, preamble)
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, text=sse(data), headers={"Content-Type": "text/event-stream"}) if stream else httpx.Response(200, json=data),
    )) as client:
        async with ModelRouter(
            provider="opencode", model="gpt-6-luna", api="responses",
            base_url="https://example.invalid/go", api_key="offline", http_client=client,
        ) as model:
            agent = Agent(name="test", model=model, output_schema=Result)
            events = [event async for event in Runner(agent=agent).run_async("user", "session", "Hola", stream=stream)]
    assert events[-1].status == "completed" and events[-1].content == '{"value":14}'
    assert events[-1].response_items[0]["phase"] == "commentary"
    if stream:
        text = "".join(event.content or "" for event in events if event.type == "stream")
        assert text == 'Voy a comprobarlo.\n\n{"value":14}'


def test_investigator_main_responses_high_rich_and_real_workflow(monkeypatch, capsys) -> None:
    path = Path(__file__).resolve().parents[1] / "ejemplos/investigador-terminal/agent.py"
    entry = runpy.run_path(str(path))["main"]
    config = {
        "MODEL": "gpt-6-luna", "OPENCODE_API": "responses",
        "GENERATION_CONFIG": {"reasoning_effort": "high"},
        "OPENCODE_BASE_URL": "https://example.invalid/go",
        "OPENCODE_API_KEY": "offline", "OPENCODE_FALLBACK_MODELS": "",
        "REQUEST_TIMEOUT": 120.0,
    }
    for name, value in config.items():
        monkeypatch.setitem(entry.__globals__, name, value)
    monkeypatch.setattr(sys, "argv", [str(path)])
    messages = iter(["Investiga energía solar", "Resume el informe anterior", "/salir"])
    monkeypatch.setattr(Console, "input", lambda *args, **kwargs: next(messages))
    requests, clients = [], []

    def handle(request):
        assert request.url.path.endswith("/openai/v1/responses")
        body = json.loads(request.content)
        requests.append(body)
        assert body["reasoning"] == {"effort": "high"} and body["store"] is False
        data = response(len(requests), tool=len(requests) == 1)
        if len(requests) == 1:
            data["output"][-1].update(name="investigar", arguments='{"tema":"energía solar"}')
        else:
            data["output"][1]["content"][0]["text"] = "Informe preparado." if len(requests) == 2 else "Resumen preparado."
        return httpx.Response(200, text=sse(data), headers={"Content-Type": "text/event-stream"})

    original = httpx.AsyncClient.__init__

    def offline_init(client, *args, **kwargs):
        clients.append(client)
        kwargs["transport"] = httpx.MockTransport(handle)
        original(client, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", offline_init)
    entry()
    output = capsys.readouterr()
    assert not output.err and "API: responses" in output.out
    assert "Tool: investigar" in output.out and "=== INFORME NAS ===" in output.out
    assert output.out.count("Informe preparado.") == 1 and output.out.count("Resumen preparado.") == 1
    assert "encrypted-" not in output.out
    assert len(requests) == 3 and clients and all(client.is_closed for client in clients)
    result = next(item for item in requests[1]["input"] if item.get("type") == "function_call_output")
    assert "=== INFORME NAS ===" in result["output"] and result["call_id"] == "call-1"
    assert any(item.get("id") == "rs-1" for item in requests[2]["input"])


def test_investigator_rejects_unknown_api_before_client_creation(monkeypatch, capsys) -> None:
    path = Path(__file__).resolve().parents[1] / "ejemplos/investigador-terminal/agent.py"
    entry = runpy.run_path(str(path))["main"]
    monkeypatch.setitem(entry.__globals__, "MODEL", "deepseek-v4.1-flash")
    monkeypatch.setitem(entry.__globals__, "OPENCODE_API", "invalid")
    monkeypatch.setattr(sys, "argv", [str(path)])
    with pytest.raises(SystemExit) as error:
        entry()
    assert error.value.code == 2 and "OPENCODE_API debe ser" in capsys.readouterr().err
