from __future__ import annotations

import json

import httpx
import pytest

from nas.adk.models import ModelRouter
from nas.adk.models.base_llm import LlmRequest

CHAT = {
    "choices": [
        {
            "message": {
                "content": "hello",
                "tool_calls": [
                    {
                        "id": "call",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"value":1}'},
                    },
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 2},
}
CHUNKS = [
    {
        "choices": [
            {
                "delta": {
                    "content": "he",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call",
                            "function": {"name": "lookup", "arguments": '{"value":'},
                        }
                    ],
                },
                "finish_reason": None,
            }
        ]
    },
    {
        "choices": [
            {
                "delta": {
                    "content": "llo",
                    "tool_calls": [{"index": 0, "function": {"arguments": "1}"}}],
                },
                "finish_reason": "tool_calls",
            }
        ]
    },
    {"choices": [], "usage": CHAT["usage"]},
]


async def test_opencode_chat_and_stream_parity() -> None:
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/openai/v1/chat/completions")
        assert request.headers["Authorization"] == "Bearer offline"
        assert request.headers["x-opencode-session"] == "session-1"
        requests.append(request)
        if json.loads(request.content).get("stream"):
            sse = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in CHUNKS) + "data: [DONE]\n\n"
            return httpx.Response(200, text=sse, headers={"Content-Type": "text/event-stream"})
        return httpx.Response(200, json=CHAT)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ModelRouter(
            provider="opencode",
            model="deepseek-v4.1-flash",
            api_key="offline",
            base_url="https://opencode.example/go",
            session_id="session-1",
            http_client=client,
        )
        request = LlmRequest(
            model="deepseek-v4.1-flash",
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {"type": "object"}},
                }
            ],
            max_tokens=30,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        full = await model.generate_content_async(request)
        stream = [chunk async for chunk in model.generate_stream_async(request)]
        assert [chunk.content for chunk in stream if chunk.type == "delta"] == ["he", "llo"]
        assert stream[-1].content == full.content == "hello"
        assert stream[-1].tool_calls == full.tool_calls
        assert CHAT["usage"].items() <= stream[-1].usage.items()
        body = json.loads(requests[0].content)
        assert body["max_tokens"] == 30 and body["temperature"] == 0.2
        assert body["tools"] == request.tools and body["response_format"] == request.response_format
        assert stream[-1].metrics["provider"] == "opencode"
        await model.aclose()
        assert not client.is_closed


@pytest.mark.parametrize(
    ("model_name", "path"),
    [
        ("deepseek-v4.1-flash", "/openai/v1/chat/completions"),
        ("gpt-6-luna", "/openai/v1/responses"),
    ],
)
async def test_opencode_auto_routes_by_model_family(model_name: str, path: str) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith(path)
        if path.endswith("/responses"):
            return httpx.Response(200, json={
                "status": "completed",
                "output": [{
                    "type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok"}],
                }],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            })
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ModelRouter(
            provider="opencode",
            model=model_name,
            api="auto",
            api_key="offline",
            base_url="https://opencode.example/go",
            http_client=client,
        )
        await model.generate_content_async(
            LlmRequest(model=model_name, messages=[{"role": "user", "content": "hi"}])
        )
