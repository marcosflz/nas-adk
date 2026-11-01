from __future__ import annotations

import json

import httpx

from nas.adk.models import ModelRouter
from nas.adk.models.base_llm import LlmRequest


def response(text: str = "ok") -> dict:
    return {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "cost": 0.0003},
    }


async def test_openrouter_headers_extra_body_and_usage() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        assert request.headers["Authorization"] == "Bearer sk-or-test"
        assert request.headers["HTTP-Referer"] == "https://app.example"
        assert request.headers["X-OpenRouter-Title"] == "NAS test"
        body = json.loads(request.content)
        assert body["provider"] == {"order": ["example"]}
        assert body["model"] == "vendor/model"
        return httpx.Response(200, json=response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ModelRouter(
            provider="openrouter",
            model="vendor/model",
            api_key="sk-or-test",
            base_url="https://openrouter.example/api/v1",
            http_referer="https://app.example",
            app_title="NAS test",
            extra_body={"provider": {"order": ["example"]}},
            http_client=client,
        )
        result = await model.generate_content_async(
            LlmRequest(model="vendor/model", messages=[{"role": "user", "content": "hi"}])
        )
    assert result.content == "ok"
    assert result.metrics["provider"] == "openrouter"
    assert result.metrics["billing"]["cost_usd"] == 0.0003


async def test_openrouter_can_use_responses_api() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/api/v1/responses")
        body = json.loads(request.content)
        assert body["store"] is False and "input" in body
        return httpx.Response(200, json={
            "status": "completed",
            "output": [{
                "type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": "ok"}],
            }],
            "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3, "cost": 0.0001},
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ModelRouter(
            provider="openrouter", model="vendor/model", api="responses",
            api_key="offline", base_url="https://openrouter.example/api/v1",
            http_client=client,
        )
        result = await model.generate_content_async(
            LlmRequest(model="vendor/model", messages=[{"role": "user", "content": "hi"}])
        )
    assert result.content == "ok"
    assert result.usage["cost"] == 0.0001


async def test_openrouter_uses_chat_completions_by_default() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(200, json=response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ModelRouter(
            provider="openrouter", model="vendor/model", api_key="offline",
            base_url="https://openrouter.example/api/v1", http_client=client,
        )
        await model.generate_content_async(
            LlmRequest(model="vendor/model", messages=[{"role": "user", "content": "hi"}])
        )
