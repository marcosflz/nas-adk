from __future__ import annotations

import io
import base64
import hashlib
import json
import zipfile
from typing import Any

import pytest

from nas.adk import Agent, Runner
from nas.adk.agents.context import Context
from nas.adk.models.base_llm import BaseLlm, LlmRequest, LlmResponse, ToolCall
from nas.adk.sessions import InMemorySessionService
from nas.adk.sessions.checkpoint import SessionCheckpoint
from nas.adk.skills import (
    DirectorySkillStore, SkillRuntime, export_zip, import_directory, import_github, import_zip,
    package_from_files,
)


def package() -> Any:
    return package_from_files({
        "SKILL.md": b"---\nname: invoices\ndescription: Process invoices\n---\nUse references/rules.md.",
        "references/rules.md": b"Amounts must include currency.",
        "scripts/report.py": b"print('ok')",
        "assets/template.bin": b"\x00\xff",
    })


def test_package_roundtrip_and_digest() -> None:
    original = package()
    assert import_zip(export_zip(original)) == [original]
    modified = original.model_dump()
    modified["instructions"] = "Pretend these are different instructions"
    with pytest.raises(ValueError, match="metadata"):
        type(original).model_validate(modified)


@pytest.mark.parametrize("path", ["../escape", "/absolute", "x/../escape", "x\\escape", "C:escape"])
def test_reject_paths(path: str) -> None:
    files = {"SKILL.md": package().read("SKILL.md"), path: b"bad"}
    with pytest.raises(ValueError):
        package_from_files(files)


def test_zip_rejects_symlinks() -> None:
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        link = zipfile.ZipInfo("invoices/link")
        link.create_system = 3
        link.external_attr = 0o120777 << 16
        archive.writestr(link, "../../etc/passwd")
    with pytest.raises(ValueError, match="Symbolic"):
        import_zip(data.getvalue())


async def test_github_uses_injected_fetcher() -> None:
    async def fetch(url: str, maximum: int) -> bytes:
        assert url == "https://codeload.github.com/team/repo/zip/main"
        assert maximum > 0
        return export_zip(package())
    assert await import_github("https://github.com/team/repo/tree/main", fetch) == [package()]
    with pytest.raises(ValueError):
        await import_github("http://localhost/repo", fetch)


async def test_github_directory_downloads_only_selected_immutable_blobs() -> None:
    skill = package()
    blobs = {}
    entries = []
    for path in skill.files:
        content = skill.read(path)
        sha = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        blobs[sha] = content
        entries.append({"path": f"skills/invoices/{path}", "mode": "100644", "type": "blob", "sha": sha, "size": len(content)})
    entries.append({"path": "unrelated/huge.bin", "mode": "100644", "type": "blob", "sha": "f" * 40, "size": 100_000_000})
    fetched = []

    async def fetch(url: str, maximum: int) -> bytes:
        fetched.append(url)
        if "/trees/" in url:
            return json.dumps({"tree": entries, "truncated": False}).encode()
        sha = url.rsplit("/", 1)[-1]
        return json.dumps({"encoding": "base64", "content": base64.b64encode(blobs[sha]).decode()}).encode()

    result = await import_github("https://github.com/team/repo/tree/main/skills/invoices", fetch)
    assert result == [skill]
    assert len(fetched) == 1 + len(skill.files)
    assert not any("f" * 40 in url for url in fetched)


def test_directory_symlink_is_not_followed(tmp_path: Any) -> None:
    root = tmp_path / "actual"
    root.mkdir()
    link = tmp_path / "invoices"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="Symbolic"):
        import_directory(link)


async def test_directory_store_is_standalone(tmp_path: Any) -> None:
    root = tmp_path / "invoices"
    for path in package().files:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(package().read(path))
    assert import_directory(root) == package()
    store = DirectorySkillStore(tmp_path)
    ctx = Context(InMemorySessionService().create_session("app", "owner"))
    runtime = SkillRuntime(store)
    refs = await store.list()
    messages = await runtime.messages(ctx)
    assert "Amounts must" not in str(messages)
    await runtime.activate_skill(refs[0].id, ctx)
    assert "Use references" in str(await runtime.messages(ctx))
    assert (await runtime.read_skill_file(refs[0].id, "references/rules.md", ctx))["text"].startswith("Amounts")
    with pytest.raises(PermissionError):
        await runtime.activate_skill("unavailable", ctx)
    with pytest.raises(PermissionError, match="isolated"):
        await runtime.run_skill_script(refs[0].id, "scripts/report.py", [], ctx)


async def test_runner_exposes_tools_and_preserves_state() -> None:
    skill = package()

    class Store:
        async def list(self) -> Any:
            return [skill.reference()]

        async def load(self, id: str) -> Any:
            return skill

    class Model(BaseLlm):
        model: str = "fake"
        seen: list[LlmRequest] = []

        async def generate_content_async(self, request: LlmRequest) -> LlmResponse:
            self.seen.append(request)
            if len(self.seen) == 1:
                assert "activate_skill" in str(request.tools)
                return LlmResponse(tool_calls=[ToolCall(id="activate", function={
                    "name": "activate_skill", "arguments": '{"skill":"invoices"}'})])
            assert "Use references" in str(request.messages)
            return LlmResponse(content="done")

    model = Model()
    svc = InMemorySessionService()
    runner = Runner(app_name="skills", agent=Agent(model=model), session_service=svc,
                    skills=SkillRuntime(Store()))
    events = [event async for event in runner.run_async("owner", "one", "Process this invoice")]
    assert events[-1].status == "completed"
    assert any(event.type == "skill" for event in events)
    session = svc.get_session("one")
    assert session is not None
    restored = InMemorySessionService().create_session("skills", "owner", "one")
    checkpoint = SessionCheckpoint.model_validate_json(
        SessionCheckpoint.from_session(session).model_dump_json())
    checkpoint.restore(restored)
    assert restored.state == session.state
    other = InMemorySessionService().create_session("skills", "other", "one")
    with pytest.raises(PermissionError):
        checkpoint.restore(other)


async def test_changed_snapshot_refuses_resume() -> None:
    skill = package()

    class Store:
        refs: list[Any] = [skill.reference()]

        async def list(self) -> Any:
            return self.refs

        async def load(self, id: str) -> Any:
            return skill

    store = Store()
    ctx = Context(InMemorySessionService().create_session("app", "user"))
    runtime = SkillRuntime(store)
    await runtime.messages(ctx)
    store.refs = []
    with pytest.raises(PermissionError, match="snapshot"):
        await runtime.messages(ctx)
