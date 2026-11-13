"""Imágenes en resultados de herramienta, para modelos que deben ver una pantalla."""

from nas.adk.models.anthropic import AnthropicLlm, to_anthropic_messages
from nas.adk.models.base_llm import LlmRequest
from nas.adk.models.openai_compat import build_chat_body, responses_input

IMAGE = {"media_type": "image/jpeg", "data": "QUJD"}
CALL = {"id": "c1", "type": "function", "function": {"name": "shot", "arguments": "{}"}}
CALL2 = {"id": "c2", "type": "function", "function": {"name": "shot", "arguments": "{}"}}


def history(*tool_messages):
    return [
        {"role": "user", "content": "mira"},
        {"role": "assistant", "content": "", "tool_calls": [CALL, CALL2][: len(tool_messages)]},
        *tool_messages,
    ]


def test_anthropic_puts_images_inside_the_tool_result():
    _, out = to_anthropic_messages(
        history({"role": "tool", "tool_call_id": "c1", "content": "ok", "images": [IMAGE]})
    )
    assert out[-1]["content"][0]["content"] == [
        {"type": "text", "text": "ok"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"}},
    ]


def test_anthropic_without_images_keeps_a_plain_string():
    _, out = to_anthropic_messages(history({"role": "tool", "tool_call_id": "c1", "content": "ok"}))
    assert out[-1]["content"][0]["content"] == "ok"


def test_anthropic_declares_images():
    assert "image" in AnthropicLlm.capabilities.input_modalities


def test_chat_sends_images_in_a_user_message_after_all_tool_results():
    messages = build_chat_body(
        LlmRequest(
            messages=history(
                {"role": "tool", "tool_call_id": "c1", "content": "a", "images": [IMAGE]},
                {"role": "tool", "tool_call_id": "c2", "content": "b"},
            )
        )
    )["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "tool", "user"]
    assert "images" not in messages[2]
    assert messages[-1]["content"] == [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}}
    ]


def test_responses_sends_images_after_the_outputs():
    items = responses_input(
        history({"role": "tool", "tool_call_id": "c1", "content": "a", "images": [IMAGE]})
    )
    assert items[-2]["type"] == "function_call_output"
    assert items[-1] == {
        "role": "user",
        "content": [{"type": "input_image", "image_url": "data:image/jpeg;base64,QUJD"}],
    }
