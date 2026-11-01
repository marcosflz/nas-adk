from __future__ import annotations

import pytest
from pydantic import ValidationError

from nas.adk.models import ModelRouter


def test_only_known_providers_are_supported() -> None:
    for provider in ("ollama", "openai", "vllm", "lmstudio", "litellm", "custom"):
        with pytest.raises(ValidationError):
            ModelRouter(provider=provider, model="deployment")


def test_provider_aliases_are_normalized() -> None:
    assert ModelRouter(provider="opencode-go", model="demo").provider == "opencode"
    assert ModelRouter(provider="jev", model="jev-latest").provider == "typesafe"
