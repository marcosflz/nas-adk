from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nas.adk.core.events.event import Event


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryConflictError(RuntimeError):
    """The memory changed after the caller's snapshot; do not overwrite it."""


class Memory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    key: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z][a-zA-Z0-9_.-]*$")
    value: str = Field(min_length=1, max_length=2000)
    category: Literal["profile", "preference", "fact"] = "fact"
    version: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None
    source_session_id: str = ""
    source_event_id: str = ""
    source_timestamp: datetime | None = None

    @field_validator("value")
    @classmethod
    def nonblank_value(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Memory values must not be blank")
        return value

    @field_validator("created_at", "updated_at", "expires_at", "source_timestamp")
    @classmethod
    def aware_dates(cls, value: datetime | None) -> datetime | None:
        if value is not None:
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("Memory timestamps must include a timezone")
            return value.astimezone(timezone.utc)
        return None

    def search_result(self) -> dict[str, object]:
        result: dict[str, object] = self.model_dump(mode="json")
        result["timestamp"] = self.updated_at.isoformat()
        return result


class MemorySnapshot(BaseModel):
    revision: int = 0
    versions: dict[str, int] = Field(default_factory=dict)


class MemoryCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1, max_length=128, pattern=r"^[a-z][a-z0-9_.-]*$")
    value: str | None = Field(default=None, min_length=1, max_length=2000)
    category: Literal["profile", "preference", "fact"] = "fact"
    action: Literal["remember", "forget"] = "remember"
    evidence: str = Field(min_length=1, max_length=2000)
    correction: bool = False

    @field_validator("value", "evidence")
    @classmethod
    def nonblank_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Memory values and evidence must not be blank")
        return value

    @model_validator(mode="after")
    def check_value(self) -> MemoryCandidate:
        if self.action == "remember" and self.value is None:
            raise ValueError("A remembered fact requires a value")
        if self.action == "forget" and self.value is not None:
            raise ValueError("A forgotten fact must not contain a value")
        return self


class MemoryExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memories: list[MemoryCandidate] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def unique_keys(self) -> MemoryExtraction:
        keys = [memory.key for memory in self.memories]
        if len(keys) != len(set(keys)):
            raise ValueError("An extraction cannot modify the same key twice")
        return self


class MemoryPolicy(BaseModel):
    """Opt-in retrieval/capture. The allowlist is not a secret detector."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    capture: Literal["manual", "auto"] = "manual"
    max_records: int = Field(default=20, ge=1, le=100)
    max_tokens: int = Field(default=1000, ge=1)
    max_candidates: int = Field(default=8, ge=1, le=10)
    max_source_chars: int = Field(default=8000, ge=1, le=32000)
    capture_timeout: float = Field(default=10, gt=0)
    delegate_write: bool = False
    allowed_keys: tuple[str, ...] = (
        "profile.name", "preferences.language", "preferences.answer_style",
    )
    allowed_prefixes: tuple[str, ...] = ("interests.",)

    def accepts(self, candidate: MemoryCandidate, source: str) -> bool:
        if candidate.evidence not in source:
            return False
        if candidate.action == "forget":
            return bool(re.search(
                r"\b(forget|delete|remove|olvida\w*|borra\w*)\b|no recuerdes",
                candidate.evidence, re.IGNORECASE,
            ))
        allowed = candidate.key in self.allowed_keys or any(
            candidate.key.startswith(prefix) for prefix in self.allowed_prefixes
        )
        # Defense in depth only: applications must choose their allowed categories.
        sensitive = re.search(
            r"password|contrase[nñ]a|api[ _-]?key|bearer\s|\bsk-[\w-]+|"
            r"-----BEGIN|secret|token\s*[:=]",
            candidate.evidence + " " + (candidate.value or ""),
            re.IGNORECASE,
        )
        return allowed and sensitive is None


class MemoryService(Protocol):
    """The existing contract, supported by both RAM and SQLite services."""

    async def add(self, user_id: str, key: str, value: str) -> None: ...
    async def get(self, user_id: str, key: str) -> str | None: ...
    async def search(self, user_id: str, query: str) -> list[dict[str, object]]: ...


@dataclass
class MemoryRunState:
    """Per-root-run state, shared with children; never stored on an Agent."""

    source: Event
    snapshot: MemorySnapshot | None = None
    retrieved: list[Memory] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    capture_suppressed: bool = False
