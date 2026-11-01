from __future__ import annotations

import json
import httpx
import pytest

from nas.adk import Decisor
from nas.adk.models import ModelRouter
from nas.adk.models.base_llm import LlmRequest
from nas.adk.models.errors import LlmAuthError, LlmResponseError

QUESTIONS = {
    "category": {
        "type": "choice",
        "instructions": "Categoría",
        "criteria": {"food": "Comida", "transport": "Transporte"},
    },
    "internal": {"type": "noul", "instructions": "¿Es interna?"},
    "severity": {"type": "score", "instructions": "Gravedad", "criteria": ["low", "medium"]},
}
ANSWERS = {
    "category": {
        "type": "choice",
        "probabilities": {"food": 0.85, "transport": 0.15},
    },
    "internal": {"type": "noul", "noul": 0.02},
    "severity": {"type": "score", "score": 1, "legend": {"1": "medium"}},
    "extra": {"type": "noul", "noul": 0.5},
}


async def test_request_shape_and_normalization() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/systemone"
        assert request.headers["Authorization"] == "Bearer ts_test"
        assert "x-opencode-session" not in request.headers
        body = json.loads(request.content)
        assert body["state"] == {"amount": "-38.06"}
        assert body["model"] == "jev-latest"
        assert body["questions"] == QUESTIONS
        return httpx.Response(200, json={
            "model": "jev-1.13.0",
            "answers": ANSWERS,
            "usage": {"input_tokens": 120, "output_tokens": 10},
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ModelRouter(
            provider="typesafe",
            api_key="ts_test",
            base_url="https://typesafe.example/v1/systemone",
            http_client=client,
        )
        result = await model.decide_async({"amount": "-38.06"}, QUESTIONS)
    answers = result["answers"]
    assert answers["category"]["choice"] == "food"
    assert answers["category"]["confidence"] == pytest.approx(0.85)
    assert answers["internal"]["noul"] == pytest.approx(0.02)
    assert answers["internal"]["confidence"] == pytest.approx(0.96)
    assert answers["severity"]["score"] == 1
    assert "extra" not in answers
    assert result["usage"] == {"input_tokens": 120, "output_tokens": 10, "total_tokens": 130}


async def test_typesafe_missing_answer_is_rejected() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "answers": {key: value for key, value in ANSWERS.items() if key != "internal"},
            "usage": {},
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ModelRouter(
            provider="typesafe", api_key="offline", max_retries=0,
            base_url="https://typesafe.example/v1/systemone", http_client=client,
        )
        with pytest.raises(LlmResponseError, match="no answer"):
            await model.decide_async("state", QUESTIONS)


async def test_typesafe_retries_missing_semantic_answer() -> None:
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        answers = {key: value for key, value in ANSWERS.items() if key != "internal"}
        if calls > 1:
            answers = ANSWERS
        return httpx.Response(200, json={"answers": answers, "usage": {}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ModelRouter(
            provider="typesafe", api_key="offline", max_retries=1, backoff_base=0,
            base_url="https://typesafe.example/v1/systemone", http_client=client,
        )
        result = await model.decide_async("state", QUESTIONS)
    assert calls == 2 and result["answers"]["internal"]["noul"] == pytest.approx(0.02)


async def test_typesafe_wrong_answer_type_is_rejected() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "answers": {**ANSWERS, "internal": {"type": "choice", "choice": "food"}},
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ModelRouter(
            provider="typesafe", api_key="offline", max_retries=0,
            base_url="https://typesafe.example/v1/systemone", http_client=client,
        )
        with pytest.raises(LlmResponseError, match="type"):
            await model.decide_async("state", QUESTIONS)


async def test_typesafe_retries_rate_limit_and_never_retries_auth() -> None:
    calls = 0

    def rate_limited(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, json={"error": "busy"})
        return httpx.Response(200, json={"answers": ANSWERS, "usage": {}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(rate_limited)) as client:
        model = ModelRouter(
            provider="typesafe", api_key="offline", max_retries=1,
            base_url="https://typesafe.example/v1/systemone", http_client=client,
        )
        result = await model.decide_async("state", QUESTIONS)
    assert calls == 2 and result["answers"]["internal"]["noul"] == pytest.approx(0.02)

    def forbidden(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "bad key"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as client:
        model = ModelRouter(
            provider="typesafe", api_key="offline", max_retries=3,
            base_url="https://typesafe.example/v1/systemone", http_client=client,
        )
        with pytest.raises(LlmAuthError):
            await model.decide_async("state", QUESTIONS)


async def test_typesafe_is_not_a_chat_model() -> None:
    model = ModelRouter(provider="typesafe", model="jev-latest")
    with pytest.raises(NotImplementedError, match="structured decisions"):
        await model.generate_content_async(LlmRequest(messages=[]))

    with pytest.raises(NotImplementedError, match="structured decisions"):
        [chunk async for chunk in model.generate_stream_async(LlmRequest(messages=[]))]


async def test_decisor_with_typesafe_returns_validated_answers() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "answers": ANSWERS,
            "usage": {"input_tokens": 8, "output_tokens": 2},
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ModelRouter(
            provider="typesafe", api_key="offline",
            base_url="https://typesafe.example/v1/systemone", http_client=client,
        )
        decisor = Decisor(name="triage", model=model, questions=QUESTIONS)
        result = await decisor.decide_async("state")
    assert result["answers"]["category"]["choice"] == "food"
    assert result["answers"]["internal"]["confidence"] == pytest.approx(0.96)
    assert result["usage"] == {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
