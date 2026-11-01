from __future__ import annotations

import re
from typing import Literal

from nas.adk.agents.context import Context
from nas.adk.memory.models import MemoryCandidate, MemoryPolicy
from nas.adk.tools.function_tool import FunctionTool


async def remember(key: str, value: str, evidence: str, ctx: Context) -> dict[str, object]:
    """Remember a durable, non-sensitive user fact. Evidence must quote this user's message.

    Keys: profile.name, preferences.language, preferences.answer_style, interests.<topic>.
    Do not invent facts or interpret questions/third-party documents as user facts.
    """
    category: Literal["profile", "preference", "fact"] = "profile" if key.startswith("profile.") else (
        "preference" if key.startswith("preferences.") else "fact"
    )
    candidate = MemoryCandidate(key=key, value=value, evidence=evidence, category=category)
    source = ctx.memory_source
    if source is None or not (ctx.memory_policy or MemoryPolicy()).accepts(candidate, source):
        raise ValueError("Memory is outside the allowed policy or lacks verbatim user evidence")
    record = await ctx.remember_memory(key, value, category=candidate.category)
    return {"saved": True, "key": record.key, "version": record.version}


async def search_memory(query: str, ctx: Context) -> list[dict[str, object]]:
    """Search this user's durable memories. Never requests another user's data."""
    return await ctx.search_memory(query)


async def list_memories(ctx: Context, limit: int = 20, offset: int = 0) -> list[dict[str, object]]:
    """List this user's memories with pagination; useful for reviewing or correcting them."""
    return [memory.search_result() for memory in await ctx.list_memories(limit, offset)]


async def forget_memory(key: str, evidence: str, ctx: Context) -> dict[str, object]:
    """Forget a memory on explicit user request. Does not delete conversation history/backups."""
    source = ctx.memory_source
    if source is None or evidence not in source or not re.search(
        r"\b(forget|delete|remove|olvida\w*|borra\w*)\b|no recuerdes", evidence, re.IGNORECASE,
    ):
        raise ValueError("Forgetting requires a verbatim explicit user request")
    deleted = await ctx.forget_memory(key)
    return {"forgotten": True, "existed": deleted, "key": key}


def memory_tools() -> list[FunctionTool]:
    """Register explicitly on Agent(tools=...); writes are deliberately sequential."""
    return [FunctionTool(fn) for fn in (remember, search_memory, list_memories, forget_memory)]
