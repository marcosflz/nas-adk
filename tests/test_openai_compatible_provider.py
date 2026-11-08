from __future__ import annotations

import json

import httpx
import pytest

from nas.adk.models import ModelRouter
from nas.adk.models.base_llm import LlmRequest
from nas.adk.models.generic import GenericOpenAILlm


def ok_response(text: str = "ok") -> dict:
    return {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


async def test_generic_provider_uses_chat_completions_at_the_given_base_url() -> None:
    seen: dict[str, object] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_response("hola"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ModelRouter(
            provider="openai-compatible",
            model="gemini-3-flash",
            base_url="https://llm.example/v1/openai/",
            api_key="sk-test",
            http_client=client,
        )
        response = await model.generate_content_async(
            LlmRequest(model="gemini-3-flash", messages=[{"role": "user", "content": "hi"}])
        )

    assert response.content == "hola"
    assert seen["url"] == "https://llm.example/v1/openai/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["model"] == "gemini-3-flash"
    assert response.usage["total_tokens"] == 5


async def test_generic_provider_works_without_api_key_for_local_servers() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert "Authorization" not in request.headers
        return httpx.Response(200, json=ok_response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ModelRouter(
            provider="openai-compatible",
            model="local-model",
            base_url="http://localhost:1234/v1",
            http_client=client,
        )
        response = await model.generate_content_async(
            LlmRequest(model="local-model", messages=[{"role": "user", "content": "hi"}])
        )
    assert response.content == "ok"


def test_generic_provider_requires_base_url_and_model() -> None:
    with pytest.raises(ValueError, match="base_url"):
        ModelRouter(provider="openai-compatible", model="x")
    with pytest.raises(ValueError, match="model"):
        ModelRouter(provider="openai-compatible", base_url="http://x/v1")


def test_router_builds_the_generic_backend() -> None:
    router = ModelRouter(provider="openai-compatible", model="m", base_url="http://x/v1")
    assert isinstance(router._get_backend(), GenericOpenAILlm)


async def test_generic_provider_is_not_a_decision_provider() -> None:
    router = ModelRouter(provider="openai-compatible", model="m", base_url="http://x/v1")
    with pytest.raises(NotImplementedError):
        await router.decide_async("s", {})
