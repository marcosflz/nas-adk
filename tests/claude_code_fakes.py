from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Callable

HELP = (
    "--tools --disallowedTools --strict-mcp-config --restricted --safe-mode "
    "--disable-slash-commands --permission-mode --permission-prompts --no-session-persistence "
    "--json-schema --output-format --system-prompt --max-turns --effort"
)

INIT = {
    "type": "system",
    "subtype": "init",
    "apiKeySource": "none",
    "tools": ["StructuredOutput"],
    "mcp_servers": [],
    "plugins": [],
}
AUTH_OK = {"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty"}


def result(structured: Any = None, **extra: Any) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "",
        "session_id": "s1",
        "usage": {"input_tokens": 10, "output_tokens": 4},
        "total_cost_usd": 0.5,
    }
    if structured is not None:
        event["structured_output"] = structured
    event.update(extra)
    return event


def final(text: str) -> dict[str, Any]:
    return result({"protocol_version": 1, "kind": "final", "text": text})


def calls(*items: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    return result(
        {
            "protocol_version": 1,
            "kind": "tool_calls",
            "calls": [{"name": n, "arguments": a} for n, a in items],
        }
    )


class FakeHandle:
    def __init__(
        self,
        lines: Sequence[Any] = (),
        *,
        code: int = 0,
        stderr: str = "",
        hang: bool = False,
    ) -> None:
        self.lines = list(lines)
        self.code = code
        self.stderr = stderr
        self.hang = hang
        self.stdin = b""
        self.terminated = 0

    async def write_stdin(self, data: bytes) -> None:
        self.stdin = data

    async def stdout_lines(self) -> AsyncIterator[bytes]:
        for line in self.lines:
            if isinstance(line, bytes):
                yield line
            else:
                yield (line if isinstance(line, str) else json.dumps(line)).encode() + b"\n"
        if self.hang:
            await asyncio.sleep(3600)

    def stderr_tail(self) -> str:
        return self.stderr

    async def terminate(self) -> None:
        self.terminated += 1

    async def wait(self) -> int:
        return self.code


class FakeTransport:
    """Responde a --version/--help/auth status y a las invocaciones principales."""

    def __init__(
        self,
        runs: Sequence[FakeHandle | Callable[[], FakeHandle]] = (),
        *,
        path: str | None = "/usr/bin/claude",
        version: str = "2.1.319 (Claude Code)",
        help_text: str = HELP,
        auth: Mapping[str, Any] | None = None,
        auth_code: int = 0,
    ) -> None:
        self.runs = list(runs)
        self.path = path
        self.version = version
        self.help_text = help_text
        self.auth = dict(AUTH_OK if auth is None else auth)
        self.auth_code = auth_code
        self.spawned: list[dict[str, Any]] = []
        self.main_handles: list[FakeHandle] = []
        self.running = 0
        self.max_running = 0

    def which(self, name: str) -> str | None:
        return self.path

    async def spawn(self, argv: Sequence[str], *, env: Mapping[str, str], cwd: str) -> FakeHandle:
        self.spawned.append({"argv": list(argv), "env": dict(env), "cwd": cwd})
        if "--version" in argv:
            return FakeHandle([self.version])
        if "--help" in argv:
            return FakeHandle([self.help_text])
        if argv[1:3] == ["auth", "status"]:
            return FakeHandle([json.dumps(self.auth)], code=self.auth_code)
        item = self.runs.pop(0)
        handle = item() if callable(item) else item
        self.main_handles.append(handle)
        return handle

    @property
    def main_calls(self) -> list[dict[str, Any]]:
        return [s for s in self.spawned if "-p" in s["argv"]]
