from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from nas.adk import Agent
from nas.adk.models.base_llm import LlmResponse
from nas.adk.services.eval import EvalCase, EvalRunner
from tests.fakes import FakeLlm


def _mock_llm(content: str = "Hello") -> AsyncMock:
    m = AsyncMock(model="test")
    m.generate_content_async = AsyncMock(return_value=LlmResponse(content=content))
    return m


@pytest.mark.asyncio
async def test_eval_runner_without_expected() -> None:
    agent = Agent(name="test", model=FakeLlm(), instruction="Say hello")
    runner = EvalRunner(agent)
    cases = [EvalCase(name="test", input="Hi")]
    with patch.object(agent, "model", _mock_llm("Hello!")):
        results = await runner.run(cases)
        assert len(results) == 1
        assert results[0].passed


@pytest.mark.asyncio
async def test_eval_runner_substring_match() -> None:
    agent = Agent(name="test", model=FakeLlm(), instruction="Say hello")
    runner = EvalRunner(agent)
    cases = [EvalCase(name="test", input="Hi", expected="Hello")]
    with patch.object(agent, "model", _mock_llm("Hello world")):
        results = await runner.run(cases)
        assert len(results) == 1
        assert results[0].passed
        assert results[0].score == 1.0


def test_eval_report() -> None:
    from nas.adk.services.eval.eval_runner import EvalResult

    results = [
        EvalResult(name="a", input="hi", output="hi", score=1.0, passed=True),
        EvalResult(name="b", input="bye", output="", score=0.0, passed=False),
    ]
    report = EvalRunner.report(results)
    assert "PASS" in report
    assert "FAIL" in report
    assert "0.50" in report


async def test_evaluation_requires_completed_response_and_rubric_uses_judge() -> None:
    from tests.test_runner_contracts import ScriptedLlm

    failed = Agent(
        name="failed",
        model=ScriptedLlm(responses=[LlmResponse(content="partial", finish_reason="length")]),
    )
    result = (await EvalRunner(failed).run([EvalCase(name="one", input="hi")]))[0]
    assert not result.passed and result.status == "error"
    agent = Agent(name="root", model=ScriptedLlm(responses=[LlmResponse(content="answer")]))
    judge_model = ScriptedLlm(responses=[LlmResponse(content='{"score":1,"reason":"good"}')])
    judge = Agent(name="judge", model=judge_model)
    result = (
        await EvalRunner(agent, judge).run([EvalCase(name="one", input="hi", rubric="Be helpful")])
    )[0]
    assert result.passed and result.score == 1 and len(judge_model.requests) == 1
    assert result.metrics["model_calls"] == 1 and "eval_seconds" in result.metrics
