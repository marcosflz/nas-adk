from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from nas.adk import Agent, Decisor, Runner
from nas.adk.models import ModelRouter
from nas.adk.models.base_llm import BaseLlm, LlmRequest, LlmResponse, ToolCall


class Ticket(BaseModel):
    message: str


RESULT = {
    "model": "decision-model",
    "answers": {
        "team": {
            "type": "choice",
            "choice": "technical",
            "confidence": 0.9,
            "probabilities": {"technical": 0.9, "billing": 0.1},
        },
    },
    "usage": {"input_tokens": 12, "output_tokens": 3},
}
QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Choose a team",
        "criteria": {"technical": "Errors", "billing": "Charges"},
    },
}


class DecisionFake(BaseLlm):
    calls: list[tuple[Any, dict[str, Any]]] = []
    result: dict[str, Any] = RESULT

    async def generate_content_async(self, request: LlmRequest) -> LlmResponse:
        raise AssertionError("Decisor must not use text generation")

    async def decide_async(
        self, state: str | dict[str, Any] | list[Any], questions: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append((state, questions))
        return self.result


class ScriptedChat(BaseLlm):
    responses: list[LlmResponse]

    async def generate_content_async(self, request: LlmRequest) -> LlmResponse:
        return self.responses.pop(0)


async def test_opencode_cannot_be_used_as_decisor() -> None:
    model = ModelRouter(provider="opencode", model="deepseek-v4.1-flash", api_key="offline")
    with pytest.raises(NotImplementedError, match="provider='openrouter'"):
        await model.decide_async("API returns 500", QUESTIONS)


async def test_decisor_runs_alone_with_structured_input_and_no_fake_stream() -> None:
    model = DecisionFake()
    decisor = Decisor(
        name="triage", model=model, questions=QUESTIONS, input_schema=Ticket,
        instruction="Classify support tickets.",
    )
    events = [
        event async for event in Runner(agent=decisor).run_async(
            "user", "session", '{"message":"API returns 500"}', stream=True,
        )
    ]

    assert events[-1].final and events[-1].status == "completed"
    assert not any(event.type == "stream" for event in events)
    assert json.loads(events[-1].content or "") == RESULT
    assert events[-1].usage == {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
    state, questions = model.calls[0]
    assert state == {"message": "API returns 500"}
    assert questions["team"]["instructions"] == "Classify support tickets.\n\nChoose a team"
    assert await decisor.decide_async({"message": "API returns 500"}) == RESULT


async def test_decisor_runs_alone_with_plain_text() -> None:
    model = DecisionFake()
    decisor = Decisor(name="triage", model=model, questions=QUESTIONS)
    events = [
        event async for event in Runner(agent=decisor).run_async(
            "user", "session", "API returns 500",
        )
    ]
    assert events[-1].status == "completed"
    assert model.calls[0][0] == "API returns 500"


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ({"type": "noul"}, {"type": "noul", "noul": 0.8}),
        ({"type": "score", "criteria": ["low", "medium", "high"]},
         {"type": "score", "score": 1.5}),
    ],
)
async def test_decisor_accepts_noul_and_score_answers(
    question: dict[str, Any], answer: dict[str, Any],
) -> None:
    result = {"answers": {"value": answer}}
    decisor = Decisor(
        name="evaluate", model=DecisionFake(result=result), questions={"value": question},
    )
    assert await decisor.decide_async("Evaluate this") == result


async def test_decisor_rejects_invalid_structured_input_before_model_call() -> None:
    model = DecisionFake()
    decisor = Decisor(
        name="triage", model=model, questions=QUESTIONS, input_schema=Ticket,
    )
    with pytest.raises(ValidationError):
        await decisor.decide_async({"message": 42})
    assert not model.calls


async def test_decisor_can_be_delegated_to_by_agent() -> None:
    decision_model = DecisionFake()
    decisor = Decisor(
        name="triage", model=decision_model, questions=QUESTIONS, input_schema=Ticket,
    )
    chat_model = ScriptedChat(
        responses=[
            LlmResponse(tool_calls=[ToolCall(
                id="call-triage",
                function={
                    "name": "request_task_triage",
                    "arguments": '{"message":"API returns 500"}',
                },
            )]),
            LlmResponse(content="Route this to technical support."),
        ],
    )
    agent = Agent(name="assistant", model=chat_model, sub_agents=[decisor])
    events = [
        event async for event in Runner(agent=agent).run_async(
            "user", "session", "Classify this issue",
        )
    ]

    assert events[-1].final and events[-1].content == "Route this to technical support."
    tool_result = next(event for event in events if event.role == "tool")
    assert json.loads(tool_result.content or "") == RESULT
    assert decision_model.calls[0][0] == {"message": "API returns 500"}


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"answers": {}},
        {"answers": {"team": {"type": "choice", "choice": "unknown"}}},
        {"answers": {"team": {"type": "choice", "choice": "technical", "confidence": 2}}},
        {**RESULT, "usage": "invalid"},
    ],
)
async def test_decisor_rejects_invalid_decision_response(result: dict[str, Any]) -> None:
    decisor = Decisor(name="triage", model=DecisionFake(result=result), questions=QUESTIONS)
    with pytest.raises(ValueError):
        await decisor.decide_async("API returns 500")


@pytest.mark.parametrize(
    "options",
    [
        {"tools": [lambda: "not supported"]},
        {"sub_agents": [object()]},
        {"generation_config": {"temperature": 0}},
        {"output_schema": Ticket},
        {"questions": {"broken": {"type": "box"}}},
    ],
)
def test_decisor_rejects_capabilities_or_questions_it_does_not_support(
    options: dict[str, Any],
) -> None:
    questions = options.pop("questions", QUESTIONS)
    with pytest.raises((ValidationError, ValueError)):
        Decisor(name="invalid", model=DecisionFake(), questions=questions, **options)
