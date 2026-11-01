from __future__ import annotations

import pytest

from nas.adk.protocols.a2a.client import A2AClient


@pytest.mark.asyncio
async def test_a2a_client_init() -> None:
    client = A2AClient("http://localhost:8010", "test-key")
    assert client._base_url == "http://localhost:8010"
    assert client._api_key == "test-key"
