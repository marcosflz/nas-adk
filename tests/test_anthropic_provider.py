from __future__ import annotations

import json
from typing import Any

import pytest

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")

from nas.adk.models import (  # noqa: E402
    LlmAuthError,
    LlmRateLimitError,
    LlmRefusalError,
    LlmRequestError,
    LlmResponseError,
    LlmTimeoutError,
    LlmTruncatedError,
    LlmUnsupportedError,
    ModelRouter,
)
from nas.adk.models.anthropic import AnthropicLlm, to_anthropic_messages  # noqa: E402
from nas.adk.models.base_llm import LlmRequest  # noqa: E402


def message(content: list[dict[str, Any]], stop: str = "end_turn", **extra: Any) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 3},
        **extra,
    }


def make(handler: Any, **kwargs: Any) -> tuple[AnthropicLlm, Any]:
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    client = anthropic.AsyncAnthropic(
        api_key="sk-test", http_client=http, max_retries=kwargs.pop("max_retries", 0)
    )
    return AnthropicLlm(model="claude-test", client=client, **kwargs), http


def req(messages: list[dict[str, Any]], **kw: Any) -> LlmRequest:
    return LlmRequest(model="claude-test", messages=messages, **kw)


def sse(events: list[tuple[str, dict[str, Any]]]) -> Any:
    body = "".join(f"event: {n}\ndata: {json.dumps(d)}\n\n" for n, d in events)
    return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})


TOOL = {
    "type": "function",
    "function": {
        "name": "consultar_gastos",
        "description": "Consulta gastos",
        "parameters": {"type": "object", "properties": {"mes": {"type": "string"}}},
    },
}


async def test_simple_text_and_system_merge() -> None:
    seen: dict[str, Any] = {}

    def handler(request: Any) -> Any:
        seen["body"] = json.loads(request.content)
        seen["version"] = request.headers.get("anthropic-version")
        return httpx2.Response(200, json=message([{"type": "text", "text": "hola"}]))

    llm, http = make(handler)
    response = await llm.generate_content_async(
        req(
            [
                {"role": "system", "content": "A"},
                {"role": "user", "content": "hi"},
                {"role": "system", "name": "nas_memory_rules", "content": "B"},
            ],
            temperature=0.3,
        )
    )
    assert response.content == "hola" and response.finish_reason == "stop"
    assert seen["body"]["system"] == "A\n\nB"
    assert "temperature" not in seen["body"]
    assert response.metrics["ignored_parameters"] == ["temperature"]
    assert response.usage["total_tokens"] == 8
    assert seen["version"] >= "2026-10-21"
    await http.aclose()


async def test_multiple_tool_calls_and_roundtrip() -> None:
    bodies: list[dict[str, Any]] = []
    thinking = {"type": "thinking", "thinking": "hmm", "signature": "sig"}

    def handler(request: Any) -> Any:
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 1:
            return httpx2.Response(
                200,
                json=message(
                    [
                        thinking,
                        {"type": "tool_use", "id": "toolu_1", "name": "consultar_gastos", "input": {"mes": "2026-11"}},
                        {"type": "tool_use", "id": "toolu_2", "name": "consultar_gastos", "input": {"mes": "2026-10"}},
                    ],
                    stop="tool_use",
                ),
            )
        return httpx2.Response(200, json=message([{"type": "text", "text": "listo"}]))

    llm, http = make(handler)
    first = await llm.generate_content_async(
        req([{"role": "user", "content": "gastos"}], tools=[TOOL])
    )
    assert first.finish_reason == "tool_calls"
    assert [c.id for c in first.tool_calls] == ["toolu_1", "toolu_2"]
    assert bodies[0]["tools"][0]["input_schema"]["type"] == "object"
    assert first.response_items[0]["content"][0]["type"] == "thinking"

    history = [
        {"role": "user", "content": "gastos"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": first.tool_calls_dicts(),
            "_response_items": first.response_items,
        },
        {"role": "tool", "tool_call_id": "toolu_2", "content": '{"total": 2}'},
        {"role": "tool", "tool_call_id": "toolu_1", "content": '{"error": "boom"}'},
    ]
    second = await llm.generate_content_async(req(history, tools=[TOOL]))
    assert second.content == "listo"
    sent = bodies[1]["messages"]
    assert sent[1]["content"][0]["type"] == "thinking"  # bloques nativos preservados
    results = sent[2]["content"]
    assert sent[2]["role"] == "user" and len(sent) == 3
    assert [r["tool_use_id"] for r in results] == ["toolu_2", "toolu_1"]
    assert results[1]["is_error"] is True and "is_error" not in results[0]
    await http.aclose()


def test_messages_rebuilt_without_response_items() -> None:
    _, msgs = to_anthropic_messages(
        [
            {"role": "user", "content": "x"},
            {
                "role": "assistant",
                "content": "pensando",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a":1}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ]
    )
    assert msgs[1]["content"] == [
        {"type": "text", "text": "pensando"},
        {"type": "tool_use", "id": "c1", "name": "f", "input": {"a": 1}},
    ]


@pytest.mark.parametrize(
    "history",
    [
        [{"role": "assistant", "content": "x"}],  # no empieza por user
        [{"role": "user", "content": "x"}, {"role": "tool", "tool_call_id": "z", "content": "r"}],
        [
            {"role": "user", "content": "x"},
            {"role": "assistant", "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "user", "content": "y"},  # falta el resultado
        ],
        [
            {"role": "user", "content": "x"},
            {"role": "assistant", "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": "{}"}}]},
        ],
    ],
)
async def test_invalid_history_makes_no_request(history: list[dict[str, Any]]) -> None:
    def handler(request: Any) -> Any:
        raise AssertionError("no debe haber red")

    llm, http = make(handler)
    with pytest.raises(LlmRequestError):
        await llm.generate_content_async(req(history))
    await http.aclose()


async def test_non_text_modality_is_rejected() -> None:
    llm, http = make(lambda r: httpx2.Response(500))
    content = [{"type": "image_url", "image_url": {"url": "x"}}]
    with pytest.raises(LlmUnsupportedError):
        await llm.generate_content_async(req([{"role": "user", "content": content}]))
    await http.aclose()


@pytest.mark.parametrize(
    ("stop", "exc"),
    [("max_tokens", LlmTruncatedError), ("refusal", LlmRefusalError), ("pause_turn", LlmResponseError)],
)
async def test_non_success_stop_reasons(stop: str, exc: type[Exception]) -> None:
    llm, http = make(
        lambda r: httpx2.Response(200, json=message([{"type": "text", "text": "par"}], stop=stop))
    )
    with pytest.raises(exc):
        await llm.generate_content_async(req([{"role": "user", "content": "x"}]))
    await http.aclose()


async def test_auth_and_rate_limit_errors_and_retry_budget() -> None:
    calls = {"n": 0}

    def unauthorized(request: Any) -> Any:
        return httpx2.Response(
            401, json={"type": "error", "error": {"type": "authentication_error", "message": "bad key"}}
        )

    llm, http = make(unauthorized)
    with pytest.raises(LlmAuthError):
        await llm.generate_content_async(req([{"role": "user", "content": "x"}]))
    await http.aclose()

    def limited(request: Any) -> Any:
        calls["n"] += 1
        return httpx2.Response(
            429,
            headers={"retry-after": "0"},
            json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}},
        )

    llm, http = make(limited, max_retries=2)
    with pytest.raises(LlmRateLimitError) as info:
        await llm.generate_content_async(req([{"role": "user", "content": "x"}]))
    assert calls["n"] == 3  # 1 + max_retries del SDK; sin reintentos duplicados del proveedor
    assert info.value.retry_after == "0"  # type: ignore[attr-defined]
    await http.aclose()


async def test_timeout_maps_to_typed_error() -> None:
    def boom(request: Any) -> Any:
        raise httpx2.ReadTimeout("t", request=request)

    llm, http = make(boom)
    with pytest.raises(LlmTimeoutError):
        await llm.generate_content_async(req([{"role": "user", "content": "x"}]))
    await http.aclose()


def stream_events(blocks: list[Any], stop: str) -> Any:
    events: list[tuple[str, dict[str, Any]]] = [
        (
            "message_start",
            {"type": "message_start", "message": message([], stop=None) | {"stop_reason": None}},  # type: ignore[arg-type]
        )
    ]
    for index, (start, deltas) in enumerate(blocks):
        events.append(("content_block_start", {"type": "content_block_start", "index": index, "content_block": start}))
        for delta in deltas:
            events.append(("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta}))
        events.append(("content_block_stop", {"type": "content_block_stop", "index": index}))
    events.append(
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop, "stop_sequence": None},
                "usage": {"output_tokens": 7},
            },
        )
    )
    events.append(("message_stop", {"type": "message_stop"}))
    return sse(events)


async def test_streaming_text_and_fragmented_tool_args() -> None:
    llm, http = make(
        lambda r: stream_events(
            [
                ({"type": "text", "text": ""}, [{"type": "text_delta", "text": "ho"}, {"type": "text_delta", "text": "la"}]),
                (
                    {"type": "tool_use", "id": "toolu_9", "name": "consultar_gastos", "input": {}},
                    [
                        {"type": "input_json_delta", "partial_json": '{"mes": "20'},
                        {"type": "input_json_delta", "partial_json": '26-11"}'},
                    ],
                ),
            ],
            "tool_use",
        )
    )
    chunks = [c async for c in llm.generate_stream_async(req([{"role": "user", "content": "x"}], tools=[TOOL]))]
    deltas = [c for c in chunks if c.type == "delta"]
    finals = [c for c in chunks if c.type == "response"]
    assert "".join(d.content or "" for d in deltas) == "hola"
    assert len(finals) == 1 and finals[0].content == "hola"  # sin duplicar texto
    assert json.loads(finals[0].tool_calls[0].function["arguments"]) == {"mes": "2026-11"}
    assert finals[0].tool_calls[0].id == "toolu_9"
    assert all(not d.tool_calls for d in deltas)  # nunca argumentos parciales
    await http.aclose()


async def test_stream_cancellation_closes_without_error() -> None:
    llm, http = make(
        lambda r: stream_events([({"type": "text", "text": ""}, [{"type": "text_delta", "text": "a"}] * 5)], "end_turn")
    )
    gen = llm.generate_stream_async(req([{"role": "user", "content": "x"}]))
    first = await gen.__anext__()
    assert first.type == "delta"
    await gen.aclose()  # no debe lanzar
    await http.aclose()


async def test_injected_client_not_closed_and_owned_client_closed() -> None:
    llm, http = make(lambda r: httpx2.Response(200, json=message([{"type": "text", "text": "x"}])))
    await llm.generate_content_async(req([{"role": "user", "content": "x"}]))
    await llm.aclose()
    assert not http.is_closed  # cliente inyectado: no es del proveedor
    await http.aclose()

    owned = AnthropicLlm(model="m", api_key="sk-x")
    client = owned._get_client()
    await owned.aclose()
    assert client._client.is_closed


async def test_missing_api_key_is_actionable() -> None:
    llm = AnthropicLlm(model="m", api_key="")
    with pytest.raises(LlmAuthError, match="ANTHROPIC_API_KEY"):
        await llm.generate_content_async(req([{"role": "user", "content": "x"}]))


async def test_output_schema_is_sanitized() -> None:
    seen: dict[str, Any] = {}

    def handler(request: Any) -> Any:
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=message([{"type": "text", "text": "{}"}]))

    llm, http = make(handler)
    schema = {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}}, "required": ["n"]}
    await llm.generate_content_async(
        req(
            [{"role": "user", "content": "x"}],
            response_format={"type": "json_schema", "json_schema": {"name": "o", "schema": schema}},
            reasoning_effort="low",
        )
    )
    out = seen["body"]["output_config"]
    assert out["effort"] == "low"
    assert out["format"]["schema"]["properties"]["n"] == {"type": "integer"}
    assert out["format"]["schema"]["additionalProperties"] is False
    await http.aclose()


async def test_decide_async_structured() -> None:
    questions = {"q1": {"type": "choice", "criteria": {"a": "x", "b": "y"}}, "q2": {"type": "noul"}}
    seen: dict[str, Any] = {}

    def handler(request: Any) -> Any:
        seen["body"] = json.loads(request.content)
        text = json.dumps({"answers": {"q1": {"type": "choice", "choice": "a"}, "q2": {"type": "noul", "noul": 0.2}}})
        return httpx2.Response(200, json=message([{"type": "text", "text": text}]))

    llm, http = make(handler)
    result = await llm.decide_async("estado", questions)
    assert result["answers"]["q1"]["choice"] == "a"
    assert "confidence" not in result["answers"]["q1"]  # no se inventa confianza
    assert result["usage"]["total_tokens"] == 8
    props = seen["body"]["output_config"]["format"]["schema"]["properties"]["answers"]["properties"]
    assert "confidence" not in props["q1"]["properties"]
    await http.aclose()


def test_router_anthropic_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-x")
    monkeypatch.setenv("NAS_MODEL_BASE_URL", "https://otro.example")
    router = ModelRouter(provider="anthropic", api_key="k")
    assert router.model == "claude-x" and not router.base_url  # no hereda NAS_MODEL_BASE_URL
    caps = router.get_capabilities()
    assert caps is not None and caps.tool_calling == "native" and not caps.personal_use_only
    with pytest.raises(Exception, match="explicit model"):
        monkeypatch.delenv("ANTHROPIC_MODEL")
        ModelRouter(provider="anthropic", api_key="k")
