from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping, Sequence

import pytest

from nas.adk.models import LlmAuthError, LlmProcessError, LlmTimeoutError
from nas.adk.models.claude_code import login
from nas.adk.models.claude_code.login import (
    SetupTokenSession,
    clean_terminal_output,
    find_login_url,
    find_token,
)

# Fragmento real (2.1.319): las palabras van separadas por movimientos de cursor y la URL
# aparece dos veces (hipervínculo OSC 8 y texto).
URL = (
    "https://claude.com/cai/oauth/authorize?code=true&client_id=9d1c250a-e61b-44d9-88ed-"
    "5944d1962f5e&response_type=code&redirect_uri=https%3A%2F%2Fplatform.claude.com%2Foauth"
    "%2Fcode%2Fcallback&scope=user%3Ainference&code_challenge=abc&code_challenge_method=S256"
    "&state=xyz"
)
BANNER = (
    "\x1b[2G\x1b[1mThis\x1b[7Gwill\x1b[12Gguide\x1b[18Gyou\x1b[22m\r\r\n"
    "\x1b[38;5;246mBrowser didn't open? Use the url below to sign in (c to copy)\x1b[39m\r\r\n"
    f"\x1b]8;id=abc;{URL}\x07\x1b[38;5;246m{URL}\x1b[39m\x1b]8;;\x07\r\r\n"
    "\x1b[2GPaste\x1b[8Gcode\x1b[13Ghere\x1b[18Gif\x1b[21Gprompted\x1b[30G>\r\r\n"
)
TOKEN = "sk-ant-oat01-" + "A1b2_C3d4-" * 9
REJECTED = (
    "  OAuth error: That code wasn't accepted. It may be incomplete, expired, already used, "
    "or from an earlier sign-in attempt.\r\n"
)


class FakePty:
    def __init__(self, before: Sequence[str], after: Sequence[str] = (), *, exit_after: bool = True, eof: bool = False) -> None:
        self.before = [c.encode() for c in before]
        self.after = [c.encode() for c in after]
        self.exit_after = exit_after
        self.eof = eof
        self.written = b""
        self.terminated = 0

    async def read(self, timeout: float) -> bytes | None:
        if self.before:
            return self.before.pop(0)
        if self.eof:
            return None
        if b"\r" in self.written:
            if self.after:
                return self.after.pop(0)
            if self.exit_after:
                return None
        await asyncio.sleep(min(timeout, 0.01))
        return b""

    async def write(self, data: bytes) -> None:
        self.written += data

    async def terminate(self) -> None:
        self.terminated += 1


def session(pty: FakePty, **kw: object) -> tuple[SetupTokenSession, list[dict[str, object]]]:
    spawned: list[dict[str, object]] = []

    async def spawn(argv: Sequence[str], env: Mapping[str, str], cwd: str) -> FakePty:
        spawned.append({"argv": list(argv), "env": dict(env), "cwd": cwd})
        return pty

    environ = {
        "PATH": "/usr/bin",
        "HOME": "/home/real",
        "DISPLAY": ":0",
        "ANTHROPIC_API_KEY": "sk-ant-api-should-not-pass",
        "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-old-token-should-not-pass",
    }
    return (
        SetupTokenSession(spawn=spawn, which=lambda _: "/usr/bin/claude", environ=environ, **kw),  # type: ignore[arg-type]
        spawned,
    )


def test_parsers_on_real_output() -> None:
    assert find_login_url(BANNER) == URL
    assert "Paste code here if prompted" in clean_terminal_output(BANNER)
    assert find_token(f"\x1b[2G{TOKEN}\x1b[39m\r\n") == TOKEN
    assert find_token("sk-ant-api03-not-a-subscription-token-xxxxxxxx") is None


async def test_full_flow_returns_token_and_cleans_up() -> None:
    pty = FakePty([BANNER[:40], BANNER[40:]], ["\x1b[2G***\r\n", f"Your token: {TOKEN}\r\n"])
    s, spawned = session(pty)
    assert await s.start() == URL
    call = spawned[0]
    env = call["env"]
    assert call["argv"] == ["/usr/bin/claude", "setup-token"]
    assert isinstance(env, dict)
    home = env["HOME"]
    assert home == call["cwd"] and home != "/home/real" and os.path.isdir(home)
    assert "DISPLAY" not in env and "ANTHROPIC_API_KEY" not in env
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env

    assert await s.submit("  the-code#state  ") == TOKEN
    assert pty.written == b"\x1b[200~the-code#state\x1b[201~\r"
    assert pty.terminated == 1 and not os.path.exists(home)


async def test_rejected_code_raises_auth_error() -> None:
    s, _ = session(FakePty([BANNER], [REJECTED], exit_after=False))
    await s.start()
    with pytest.raises(LlmAuthError, match="wasn't accepted"):
        await s.submit("bad#code")


async def test_exit_without_token_is_auth_error() -> None:
    s, _ = session(FakePty([BANNER], []))
    await s.start()
    with pytest.raises(LlmAuthError):
        await s.submit("code")


@pytest.mark.parametrize("code", ["", "with space", "a\x1b[201~b", "x" * 600])
async def test_invalid_codes_are_rejected_before_writing(code: str) -> None:
    pty = FakePty([BANNER])
    s, _ = session(pty)
    await s.start()
    with pytest.raises(LlmAuthError):
        await s.submit(code)
    assert pty.written == b""
    await s.close()


async def test_no_url_closes_and_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    pty = FakePty(["Something unexpected\r\n"], eof=True)
    s, spawned = session(pty)
    with pytest.raises(LlmProcessError, match="sign-in URL"):
        await s.start()
    env = spawned[0]["env"]
    assert isinstance(env, dict)
    assert pty.terminated == 1 and not os.path.exists(env["HOME"])


async def test_url_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(login, "URL_TIMEOUT_SECONDS", 0.05)
    s, _ = session(FakePty([], exit_after=False))
    with pytest.raises(LlmTimeoutError):
        await s.start()


async def test_expired_session_and_double_submit() -> None:
    s, _ = session(FakePty([BANNER]), ttl=0)
    await s.start()
    with pytest.raises(LlmTimeoutError):
        await s.submit("code")

    s, _ = session(FakePty([BANNER], [TOKEN + "\r\n"]))
    await s.start()
    await s.submit("code")
    with pytest.raises(LlmProcessError):
        await s.submit("code")


async def test_missing_executable() -> None:
    s = SetupTokenSession(which=lambda _: None, environ={})
    with pytest.raises(LlmProcessError, match="not found"):
        await s.start()
