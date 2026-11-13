"""El razonamiento que publica el proveedor llega como deltas aparte, sin mezclarse con el texto."""

from nas.adk.models.anthropic import summarized_thinking
from nas.adk.models.base_llm import LlmRequest
from nas.adk.models.openai_compat import parse_chat_stream, parse_responses_stream


async def _aiter(items):
    for item in items:
        yield item


def _request(**kw):
    return LlmRequest(model="m", messages=[{"role": "user", "content": "hola"}], **kw)


def test_summary_is_only_requested_when_asked_and_where_thinking_is_already_on():
    asked = _request(reasoning_summary=True)
    assert summarized_thinking("claude-opus-5-5", asked) == {"type": "adaptive", "display": "summarized"}
    assert summarized_thinking("claude-sonnet-5", asked) == {"type": "adaptive", "display": "summarized"}
    # En modelos anteriores activarlo cambiaría el coste: no se toca.
    assert summarized_thinking("claude-opus-4-7", asked) is None
    assert summarized_thinking("claude-haiku-4-5", asked) is None
    assert summarized_thinking("claude-opus-5-5", _request()) is None


async def test_chat_stream_separates_reasoning_from_text():
    chunks = [
        {"choices": [{"delta": {"reasoning_content": "pienso"}}]},
        {"choices": [{"delta": {"reasoning": " más"}}]},
        {"choices": [{"delta": {"content": "hola"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    out = [r async for r in parse_chat_stream(_aiter(chunks))]
    assert [r.reasoning for r in out if r.type == "delta" and r.reasoning] == ["pienso", " más"]
    assert [r.content for r in out if r.type == "delta" and r.content] == ["hola"]
    assert out[-1].type == "response" and out[-1].content == "hola" and out[-1].reasoning is None


async def test_responses_stream_surfaces_summary_deltas():
    done = {"type": "response.completed", "response": {"output": [], "status": "completed"}}
    events = [{"type": "response.reasoning_summary_text.delta", "delta": "resumen"}, done]
    out = [r async for r in parse_responses_stream(_aiter(events))]
    assert out[0].reasoning == "resumen" and out[0].content is None
