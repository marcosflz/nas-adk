from __future__ import annotations

import base64
import json
import time
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from nas.adk.models import LlmAuthError, LlmPermissionError, LlmQuotaError, LlmRateLimitError
from nas.adk.models.base_llm import LlmRequest
from nas.adk.models.chatgpt import ChatGptPlanLlm
from nas.adk.models.chatgpt import oauth
from nas.adk.models.chatgpt.llm import TOOL_NAMESPACE, parse_models, plan_body

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "buscar",
            "description": "Busca",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }
]


def sse(*events: dict[str, Any]) -> bytes:
    return b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)


def completed(output: list[dict[str, Any]], **usage: int) -> dict[str, Any]:
    return {
        "type": "response.completed",
        "response": {
            "status": "completed",
            "output": output,
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7, **usage},
        },
    }


def model(handler: Any) -> ChatGptPlanLlm:
    return ChatGptPlanLlm(
        model="gpt-6.1-sol",
        access_token="at-123",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        backoff_base=0,
    )


def test_plan_body_follows_the_route_rules() -> None:
    body = plan_body(
        {
            "model": "m",
            "temperature": 0.3,
            "max_output_tokens": 10,
            "input": [
                {"role": "system", "content": "Eres útil"},
                {"role": "user", "content": "hola"},
                {"type": "function_call", "call_id": "c1", "name": "buscar", "arguments": "{}"},
            ],
            "tools": [{"type": "function", "strict": True, "name": "buscar", "parameters": {}}],
        }
    )
    assert body["instructions"] == "Eres útil"
    assert "temperature" not in body and "max_output_tokens" not in body
    assert body["store"] is False and body["stream"] is True
    assert body["input"][0] == {"role": "user", "content": "hola"}
    assert body["input"][1]["namespace"] == TOOL_NAMESPACE
    namespace = body["tools"][0]
    assert namespace["type"] == "namespace" and namespace["name"] == TOOL_NAMESPACE
    assert namespace["tools"][0]["name"] == "buscar" and namespace["tools"][0]["strict"] is False


async def test_streams_text_and_returns_tool_calls() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=sse(
                {"type": "response.output_text.delta", "item_id": "m1", "delta": "Ho"},
                {"type": "response.output_text.delta", "item_id": "m1", "delta": "la"},
                completed(
                    [
                        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hola"}]},
                        {
                            "type": "function_call",
                            "call_id": "c1",
                            "name": "buscar",
                            "namespace": TOOL_NAMESPACE,
                            "arguments": '{"q":"x"}',
                        },
                    ]
                ),
            ),
            headers={"content-type": "text/event-stream"},
        )

    llm = model(handler)
    request = LlmRequest(
        messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "hola"}],
        tools=TOOLS,
        temperature=0.2,
    )
    chunks = [c async for c in llm.generate_stream_async(request)]
    assert "".join(c.content or "" for c in chunks if c.type == "delta") == "Hola"
    final = chunks[-1]
    assert final.tool_calls[0].function["name"] == "buscar" and final.usage["total_tokens"] == 7
    assert seen["url"] == "https://api.openai.com/v1/responses"
    assert seen["auth"] == "Bearer at-123"
    assert seen["body"]["instructions"] == "S" and seen["body"]["stream"] is True
    assert "temperature" not in seen["body"]
    # La respuesta no-stream consume el stream.
    assert (await model(handler).generate_content_async(request)).content == "Hola"


@pytest.mark.parametrize(
    ("status", "code", "error"),
    [
        (429, "subscription_sharing_usage_limit_exceeded", LlmQuotaError),
        (503, "subscription_sharing_usage_unavailable", LlmRateLimitError),
        (401, "subscription_sharing_invalid_user", LlmAuthError),
        (403, "subscription_sharing_user_not_eligible", LlmPermissionError),
    ],
)
async def test_plan_error_codes_are_typed(status: int, code: str, error: type) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"code": code, "message": "x"}})

    llm = model(handler)
    llm.max_retries = 0
    with pytest.raises(error):
        await llm.generate_content_async(LlmRequest(messages=[{"role": "user", "content": "hola"}]))


async def test_failed_event_with_usage_limit_is_a_quota_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse(
                {
                    "type": "response.failed",
                    "response": {
                        "status": "failed",
                        "error": {"code": "subscription_sharing_usage_limit_exceeded", "message": "límite"},
                    },
                }
            ),
        )

    llm = model(handler)
    llm.max_retries = 0
    with pytest.raises(LlmQuotaError):
        await llm.generate_content_async(LlmRequest(messages=[{"role": "user", "content": "hola"}]))


def test_parse_models_keeps_listed_in_order() -> None:
    data = {
        "models": [
            {"slug": "gpt-6.1-sol", "display_name": "GPT-6.1 Sol", "visibility": "list"},
            {"slug": "hidden", "display_name": "Hidden", "visibility": "hide"},
            {"slug": "gpt-mini", "visibility": "list"},
        ]
    }
    assert parse_models(data) == [
        {"id": "gpt-6.1-sol", "label": "GPT-6.1 Sol"},
        {"id": "gpt-mini", "label": "gpt-mini"},
    ]


def id_token(**claims: Any) -> str:
    def part(data: dict[str, Any]) -> str:
        return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()

    return f"{part({'alg': 'RS256'})}.{part(claims)}.sig"


async def test_login_exchange_refresh_and_models() -> None:
    pending = oauth.start_login(host_id="urn:uuid:host", app_name="Nexus")
    query = parse_qs(urlsplit(pending.url).query)
    assert query["client_id"] == ["dynamic_agent_client"]
    assert query["redirect_uri"] == [oauth.REDIRECT_URI]
    assert query["code_challenge_method"] == ["S256"] and "chatgpt.tokens.use.direct" in query["scope"][0]
    assert query["ext_agent_host_id"] == ["urn:uuid:host"] and query["agent_name_hint"] == ["Nexus"]

    forms: list[dict[str, list[str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            assert request.headers["authorization"] == "Bearer at-2"
            return httpx.Response(200, json={"models": [{"slug": "m", "display_name": "M", "visibility": "list"}]})
        form = parse_qs(request.content.decode())
        forms.append(form)
        if form["grant_type"] == ["authorization_code"]:
            return httpx.Response(
                200,
                json={
                    "access_token": "at-1",
                    "refresh_token": "rt-1",
                    "expires_in": 3600,
                    "scope": oauth.SCOPE,
                    "id_token": id_token(nonce=pending.nonce, exp=time.time() + 60, email="ana@example.com"),
                },
            )
        return httpx.Response(200, json={"access_token": "at-2", "refresh_token": "rt-2", "expires_in": 3600})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    pasted = f"{oauth.REDIRECT_URI}?code=abc&state={pending.state}&scope=openid+chatgpt.tokens.use.direct&client_id=issued-1"
    creds = await oauth.exchange_code(pasted, pending, client=client)
    assert (creds.client_id, creds.access_token, creds.refresh_token, creds.account) == (
        "issued-1",
        "at-1",
        "rt-1",
        "ana@example.com",
    )
    assert forms[0]["code_verifier"] == [pending.verifier] and forms[0]["client_id"] == ["issued-1"]

    renewed = await oauth.refresh(creds, client=client)
    assert renewed.refresh_token == "rt-2" and renewed.client_id == "issued-1"
    assert forms[1]["refresh_token"] == ["rt-1"] and "scope" not in forms[1]
    assert oauth.ChatGptCredentials.from_json(renewed.to_json()) == renewed
    assert await oauth.list_models("at-2", client=client) == [{"id": "m", "label": "M"}]


def test_redirect_validation() -> None:
    pending = oauth.start_login(host_id="h", app_name="Nexus")
    with pytest.raises(LlmAuthError, match="another sign-in"):
        oauth.parse_redirect(f"{oauth.REDIRECT_URI}?code=a&state=other", pending)
    with pytest.raises(LlmAuthError, match="127.0.0.1"):
        oauth.parse_redirect(f"https://evil.example/auth/callback?code=a&state={pending.state}", pending)
    with pytest.raises(LlmPermissionError):
        oauth.parse_redirect(f"{oauth.REDIRECT_URI}?error=access_denied&state={pending.state}", pending)
    with pytest.raises(LlmPermissionError):
        oauth.parse_redirect(f"{oauth.REDIRECT_URI}?code=a&state={pending.state}&scope=openid", pending)
    # Con client_id ya emitido no se vuelve a registrar.
    again = oauth.start_login(host_id="h", app_name="Nexus", client_id="issued-1")
    assert "agent_name_hint" not in again.url and "client_id=issued-1" in again.url


async def test_terminal_refresh_error_means_sign_in_again() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "refresh_token_reused"})

    creds = oauth.ChatGptCredentials("c", "a", "r", time.time())
    with pytest.raises(LlmAuthError):
        await oauth.refresh(creds, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def test_nonce_mismatch_is_rejected() -> None:
    pending = oauth.start_login(host_id="h", app_name="Nexus")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"access_token": "a", "refresh_token": "r", "id_token": id_token(nonce="other", exp=time.time() + 60)},
        )

    with pytest.raises(LlmAuthError, match="does not match"):
        await oauth.exchange_code(
            f"{oauth.REDIRECT_URI}?code=a&state={pending.state}",
            pending,
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
