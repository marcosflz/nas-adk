from __future__ import annotations

import json

import httpx
import pytest

from nas.adk.models import ModelRouter
from nas.adk.models.base_llm import LlmRequest
from nas.adk.models.errors import LlmAuthError


def chat_response(text: str = "ok") -> dict:
    return {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "opencode": {"allowance": 0.001, "list_cost": 0.0012, "billing_source": "go"},
        },
    }


async def test_opencode_headers_usage_and_session() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/openai/v1/chat/completions")
        assert request.headers["Authorization"] == "Bearer oc_sk_test"
        assert request.headers["User-Agent"].startswith("nas-adk/")
        assert request.headers["x-opencode-session"] == "session-42"
        return httpx.Response(200, json=chat_response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ModelRouter(
            provider="opencode",
            model="deepseek-v4.1-flash",
            base_url="https://opencode.example/go",
            api_key="oc_sk_test",
            http_client=client,
        )
        response = await model.generate_content_async(
            LlmRequest(
                model="deepseek-v4.1-flash",
                session_id="session-42",
                messages=[{"role": "user", "content": "hi"}],
            )
        )
    assert response.content == "ok"
    assert response.metrics["billing"]["allowance_usd"] == 0.001
    assert response.metrics["billing"]["source"] == "go"


async def test_runner_session_id_reaches_opencode_header() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-opencode-session"] == "runner-session"
        return httpx.Response(200, json=chat_response("ok"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ModelRouter(
            provider="opencode", model="demo",
            base_url="https://opencode.example/go", api_key="offline", http_client=client,
        )
        from nas.adk import Agent, Runner

        events = [
            event
            async for event in Runner(agent=Agent(name="test", model=model)).run_async(
                "user", "runner-session", "Hola",
            )
        ]
    assert events[-1].content == "ok"


async def test_opencode_falls_back_on_rate_limit_but_not_on_auth() -> None:
    calls: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        calls.append(model)
        if model == "primary":
            return httpx.Response(429, json={"error": "quota"})
        return httpx.Response(200, json=chat_response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ModelRouter(
            provider="opencode", model="primary", fallback_models=("backup",),
            base_url="https://opencode.example/go", api_key="offline", max_retries=0,
            http_client=client,
        )
        response = await model.generate_content_async(
            LlmRequest(model="primary", messages=[{"role": "user", "content": "hi"}])
        )
    assert calls == ["primary", "backup"]
    assert response.metrics["fallback_used"] is True
    assert response.metrics["attempted_models"] == ["primary", "backup"]

    def unauthorized(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(unauthorized)) as client:
        model = ModelRouter(
            provider="opencode", model="primary", fallback_models=("backup",),
            base_url="https://opencode.example/go", api_key="offline", http_client=client,
        )
        with pytest.raises(LlmAuthError):
            await model.generate_content_async(
                LlmRequest(model="primary", messages=[{"role": "user", "content": "hi"}])
            )


async def test_opencode_stream_fallback_before_first_delta() -> None:
    calls: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        calls.append(model)
        if model == "primary":
            return httpx.Response(429, json={"error": "quota"})
        chunks = [
            {"choices": [{"delta": {"content": "hola"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        return httpx.Response(
            200,
            text="".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n",
            headers={"Content-Type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ModelRouter(
            provider="opencode", model="primary", fallback_models=("backup",),
            base_url="https://opencode.example/go", api_key="offline",
            max_retries=0, http_client=client,
        )
        stream = [
            chunk
            async for chunk in model.generate_stream_async(
                LlmRequest(model="primary", messages=[{"role": "user", "content": "hi"}])
            )
        ]
    assert calls == ["primary", "backup"]
    assert [chunk.content for chunk in stream if chunk.type == "delta"] == ["hola"]
    assert stream[-1].metrics["fallback_used"] is True


async def test_opencode_stream_never_retries_after_first_delta() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        # Falta el terminal: el parser detecta un stream incompleto tras el delta.
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"parcial"},"finish_reason":null}]}\n\n',
            headers={"Content-Type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ModelRouter(
            provider="opencode", model="primary", fallback_models=("backup",),
            base_url="https://opencode.example/go", api_key="offline",
            max_retries=2, http_client=client,
        )
        received = []
        with pytest.raises(Exception):
            async for chunk in model.generate_stream_async(
                LlmRequest(model="primary", messages=[{"role": "user", "content": "hi"}])
            ):
                received.append(chunk)
    assert [chunk.content for chunk in received] == ["parcial"]


async def test_opencode_decide_is_explicitly_unsupported() -> None:
    model = ModelRouter(provider="opencode", model="demo", api_key="offline")
    with pytest.raises(NotImplementedError, match="provider='typesafe'"):
        await model.decide_async("state", {"q": {"type": "noul"}})


async def test_opencode_retries_transient_errors() -> None:
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(503, json={"error": "temporary"})
        return httpx.Response(200, json=chat_response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ModelRouter(
            provider="opencode", model="demo", base_url="https://opencode.example/go",
            api_key="offline", max_retries=2, backoff_base=0, http_client=client,
        )
        response = await model.generate_content_async(
            LlmRequest(model="demo", messages=[{"role": "user", "content": "hi"}])
        )
    assert calls == 3 and response.content == "ok"
