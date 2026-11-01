from __future__ import annotations

import pytest

from nas.adk.services.artifacts.in_memory_artifact_service import (
    InMemoryArtifactService,
)


@pytest.mark.asyncio
async def test_save_and_get_artifact() -> None:
    svc = InMemoryArtifactService()
    art = await svc.save("sess1", "test.txt", b"hello world")
    assert art.name == "test.txt"
    assert art.content == b"hello world"
    retrieved = await svc.get(art.id)
    assert retrieved is not None
    assert retrieved.content == b"hello world"


@pytest.mark.asyncio
async def test_list_by_session() -> None:
    svc = InMemoryArtifactService()
    await svc.save("sess1", "a.txt", b"aaa")
    await svc.save("sess1", "b.txt", b"bbb")
    await svc.save("sess2", "c.txt", b"ccc")
    items = await svc.list_by_session("sess1")
    assert len(items) == 2
