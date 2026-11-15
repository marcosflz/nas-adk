"""Portable Agent Skills contracts. Packages contain base64 encoded file bytes."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_PACKAGE_BYTES = 10 * 1024 * 1024
MAX_FILES = 256


def safe_skill_path(value: str) -> str:
    path = PurePosixPath(value)
    if (not value or "\\" in value or "\x00" in value or path.is_absolute()
            or any(part in {"", ".", ".."} for part in value.split("/"))
            or ":" in value or len(value) > 512):
        raise ValueError("Skill paths must be relative paths inside the package")
    return value


class SkillManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=1024)
    license: str | None = None
    compatibility: str | None = Field(default=None, max_length=500)
    metadata: dict[str, str] = Field(default_factory=dict)
    allowedTools: str | None = None

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value):
            raise ValueError("Use lowercase letters, numbers and single hyphens for skill names")
        return value

    @field_validator("description")
    @classmethod
    def nonempty_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("A skill description cannot be blank")
        return value


class SkillRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    name: str
    description: str
    digest: str = Field(pattern=r"^[a-f0-9]{64}$")


class SkillPackage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest: SkillManifest
    instructions: str
    files: dict[str, str]
    digest: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def check_content(self) -> SkillPackage:
        if not 1 <= len(self.files) <= MAX_FILES or "SKILL.md" not in self.files:
            raise ValueError("A skill needs SKILL.md and at most 256 files")
        total = 0
        for path, content in self.files.items():
            safe_skill_path(path)
            total += len(base64.b64decode(content, validate=True))
            if total > MAX_PACKAGE_BYTES:
                raise ValueError("Skill package exceeds 10 MB")
        if package_digest(self.files) != self.digest:
            raise ValueError("Skill package digest does not match its files")
        # Metadata and instructions must also be derived from these exact bytes.
        from .packages import parse_skill_md
        manifest, instructions = parse_skill_md(self.read("SKILL.md").decode("utf-8"))
        if manifest != self.manifest or instructions != self.instructions:
            raise ValueError("Skill metadata does not match SKILL.md")
        return self

    def read(self, path: str) -> bytes:
        return base64.b64decode(self.files[safe_skill_path(path)], validate=True)

    def reference(self, id: str | None = None) -> SkillRef:
        return SkillRef(id=id or f"{self.manifest.name}@{self.digest}",
                        name=self.manifest.name, description=self.manifest.description,
                        digest=self.digest)


def package_digest(files: dict[str, str]) -> str:
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def skills_json_schema() -> dict[str, Any]:
    return SkillPackage.model_json_schema()
