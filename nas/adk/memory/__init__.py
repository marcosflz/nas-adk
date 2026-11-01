from __future__ import annotations

from typing import TYPE_CHECKING

from nas.adk.memory.in_memory_memory_service import (
    InMemoryMemoryService,
    MemoryRecord,
)
from nas.adk.memory.extractor import MemoryExtractor
from nas.adk.memory.models import (
    Memory, MemoryCandidate, MemoryConflictError, MemoryPolicy, MemoryService,
)
from nas.adk.memory.sqlite_memory_service import SQLiteMemoryService

if TYPE_CHECKING:
    from nas.adk.tools.function_tool import FunctionTool


def memory_tools() -> list[FunctionTool]:
    # Import lazily: FunctionTool depends on Context, which imports memory models.
    from nas.adk.memory.tools import memory_tools as build

    return build()

__all__ = [
    "InMemoryMemoryService",
    "MemoryRecord",
    "Memory",
    "MemoryCandidate",
    "MemoryConflictError",
    "MemoryPolicy",
    "MemoryService",
    "MemoryExtractor",
    "SQLiteMemoryService",
    "memory_tools",
]
