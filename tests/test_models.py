from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import ValidationError

from nas.adk import Agent
from nas.adk.models import ModelRouter
from tests.fakes import FakeLlm


def test_model_router_configuration_is_explicit_and_hides_api_key() -> None:
    model = ModelRouter(
        provider="opencode",
        base_url="https://opencode.example/go",
        api_key="offline-secret",
        model="deepseek-v4.1-flash",
    )
    assert model.model == "deepseek-v4.1-flash"
    assert model.provider == "opencode"
    assert model.model_dump()["base_url"] == "https://opencode.example/go"
    assert "offline-secret" not in repr(model)
    assert "offline-secret" not in str(model.model_dump())


def test_model_router_normalizes_aliases_and_provider_defaults() -> None:
    assert ModelRouter(provider="jev", model="jev-latest").provider == "typesafe"
    assert ModelRouter(provider="opencode-go", model="demo").provider == "opencode"
    assert ModelRouter(provider="openrouter", model="slug").api == "chat_completions"
    assert ModelRouter(provider="typesafe").model == "jev-latest"
    assert ModelRouter(provider="opencode").model == "deepseek-v4.1-flash"


def test_model_router_rejects_unknown_and_removed_providers() -> None:
    with pytest.raises(ValidationError):
        ModelRouter(provider="ollama", model="local")
    with pytest.raises(ValidationError, match="Foundry support was removed"):
        ModelRouter(provider="foundry", model="deployment")
    with pytest.raises(ValidationError, match="requires an explicit model"):
        ModelRouter(provider="openrouter")


def test_model_router_rejects_decision_outside_openrouter() -> None:
    from nas.adk.models import DecisionOptions

    with pytest.raises(ValidationError, match="only valid"):
        ModelRouter(provider="opencode", model="demo", decision=DecisionOptions())
    with pytest.raises(ValidationError, match="only valid"):
        ModelRouter(provider="typesafe", decision=DecisionOptions())
    with pytest.raises(ValidationError, match="fallback"):
        ModelRouter(provider="typesafe", fallback_models=("other",))
    with pytest.raises(ValidationError, match="api"):
        ModelRouter(provider="typesafe", api="responses")


def test_model_router_reads_provider_environment(monkeypatch) -> None:
    monkeypatch.setenv("OPENCODE_BASE_URL", "https://opencode.env/go")
    monkeypatch.setenv("OPENCODE_MODEL", "env-model")
    monkeypatch.setenv("OPENCODE_API", "responses")
    monkeypatch.setenv("OPENCODE_FALLBACK_MODELS", "one, two")
    monkeypatch.setenv("OPENCODE_TIMEOUT", "33")
    monkeypatch.setenv("OPENCODE_MAX_RETRIES", "7")
    model = ModelRouter(provider="opencode")
    assert model.model == "env-model"
    assert model.base_url == "https://opencode.env/go"
    assert model.api == "responses"
    assert model.fallback_models == ("one", "two")
    assert model.timeout == 33 and model.max_retries == 7

    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.env/api/v1")
    monkeypatch.setenv("OPENROUTER_MODEL", "env/vendor")
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://app.example")
    monkeypatch.setenv("OPENROUTER_APP_TITLE", "Env app")
    openrouter = ModelRouter(provider="openrouter")
    assert openrouter.model == "env/vendor"
    assert openrouter.base_url == "https://openrouter.env/api/v1"
    assert openrouter.api == "chat_completions"
    assert openrouter.http_referer == "https://app.example"
    assert openrouter.app_title == "Env app"


def test_agent_propagates_session_id_to_request() -> None:
    from types import SimpleNamespace

    agent = Agent(name="test", model=FakeLlm())
    request = agent.prepare_request(
        [{"role": "user", "content": "Hola"}],
        SimpleNamespace(session=SimpleNamespace(id="session-123")),
    )
    assert request.session_id == "session-123"
    assert agent.prepare_request([{"role": "user", "content": "Hola"}]).session_id is None


async def test_agents_do_not_close_a_shared_model_unless_requested() -> None:
    model = FakeLlm()
    child = Agent(name="child", model=model)
    root = Agent(name="root", model=model, sub_agents=[child])
    with patch.object(FakeLlm, "aclose", new_callable=AsyncMock) as close:
        await root.aclose()
        close.assert_not_awaited()
        await root.aclose_models()
        close.assert_awaited_once()


async def test_router_does_not_close_an_injected_http_client() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    ) as client:
        model = ModelRouter(provider="opencode", model="demo", api_key="offline", http_client=client)
        await model.aclose()
        assert not client.is_closed
