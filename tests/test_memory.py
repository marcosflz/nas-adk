from __future__ import annotations

import pytest

from nas.adk.memory.in_memory_memory_service import (
    InMemoryMemoryService,
)


@pytest.mark.asyncio
async def test_memory_add_and_get() -> None:
    svc = InMemoryMemoryService()
    await svc.add("user1", "name", "Alice")
    result = await svc.get("user1", "name")
    assert result == "Alice"


@pytest.mark.asyncio
async def test_memory_get_missing() -> None:
    svc = InMemoryMemoryService()
    result = await svc.get("user1", "nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_memory_scoped_to_user() -> None:
    svc = InMemoryMemoryService()
    await svc.add("user1", "name", "Alice")
    await svc.add("user2", "name", "Bob")
    assert await svc.get("user1", "name") == "Alice"
    assert await svc.get("user2", "name") == "Bob"


@pytest.mark.asyncio
async def test_memory_search() -> None:
    svc = InMemoryMemoryService()
    await svc.add("user1", "city", "Madrid")
    await svc.add("user1", "language", "Spanish")
    results = await svc.search("user1", "Madrid")
    assert len(results) >= 1
    assert results[0]["key"] == "city"
