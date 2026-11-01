from __future__ import annotations

import json
from typing import Any

import httpx

from nas.adk import Decisor
from nas.adk.models import DecisionOptions, ModelRouter

QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Choose a team",
        "criteria": {"technical": "Errors", "billing": "Charges"},
    },
    "urgent": {"type": "noul", "instructions": "Is it urgent?"},
    "severity": {
        "type": "score",
        "instructions": "How severe?",
        "criteria": ["low", "medium", "high"],
    },
}
ANSWERS = {
    "team": {"type": "choice", "choice": "technical", "confidence": 0.9},
    "urgent": {"type": "noul", "noul": 0.2},
    "severity": {"type": "score", "score": 2, "confidence": 0.8},
}


def _completion(text: str) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    }


async def test_openrouter_decision_uses_json_schema_and_provider_routing() -> None:
    requests: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        assert request.url.path.endswith("/chat/completions")
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True
        schema = body["response_format"]["json_schema"]["schema"]
        assert schema["properties"]["answers"]["required"] == ["team", "urgent", "severity"]
        assert schema["properties"]["answers"]["properties"]["team"]["properties"]["choice"]["enum"] == [
            "technical", "billing",
        ]
        assert body["provider"] == {"require_parameters": True}
        assert body["temperature"] == 0
        return httpx.Response(200, json=_completion(json.dumps({"answers": ANSWERS})))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ModelRouter(
            provider="openrouter",
            model="vendor/model",
            api_key="offline",
            base_url="https://openrouter.example/api/v1",
            decision=DecisionOptions(strict=True, require_parameters=True),
            http_client=client,
        )
        result = await model.decide_async({"message": "API down"}, QUESTIONS)
    assert result["answers"] == ANSWERS
    assert result["usage"] == {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
    assert len(requests) == 1


async def test_openrouter_decision_falls_back_from_json_schema_to_json_object() -> None:
    requests: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            assert body["response_format"]["type"] == "json_schema"
            return httpx.Response(400, json={"error": {"message": "response_format unsupported"}})
        assert body["response_format"] == {"type": "json_object"}
        return httpx.Response(200, json=_completion("```json\n" + json.dumps({"answers": ANSWERS}) + "\n```"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ModelRouter(
            provider="openrouter",
            model="vendor/model",
            api_key="offline",
            base_url="https://openrouter.example/api/v1",
            decision=DecisionOptions(max_retries=1),
            http_client=client,
        )
        result = await model.decide_async("state", QUESTIONS)
    assert result["answers"] == ANSWERS
    assert len(requests) == 2


async def test_openrouter_decision_retries_invalid_json() -> None:
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, json=_completion("not json"))
        return httpx.Response(200, json=_completion(json.dumps({"answers": ANSWERS})))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ModelRouter(
            provider="openrouter",
            model="vendor/model",
            api_key="offline",
            base_url="https://openrouter.example/api/v1",
            decision=DecisionOptions(max_retries=1),
            http_client=client,
        )
        result = await model.decide_async("state", QUESTIONS)
    assert result["answers"] == ANSWERS and calls == 2


async def test_decisor_with_openrouter_model_returns_validated_answers() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion(json.dumps({"answers": ANSWERS})))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ModelRouter(
            provider="openrouter",
            model="vendor/model",
            api_key="offline",
            base_url="https://openrouter.example/api/v1",
            decision=DecisionOptions(max_retries=1),
            http_client=client,
        )
        decisor = Decisor(name="triage", model=model, questions=QUESTIONS)
        result = await decisor.decide_async({"message": "API down"})
    assert result["answers"] == ANSWERS
    assert result["usage"] == {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
