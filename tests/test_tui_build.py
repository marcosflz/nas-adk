from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

pytest.importorskip("textual")

from nas.adk.tui.build import build_exe, main


@pytest.mark.parametrize("directory,onefile,name", [
    (False, True, None), (True, True, None),
    (False, False, "Mi Asistente"), (True, True, "MiAsistente.exe"),
])
def test_build_arguments_resources_and_no_entry_execution(tmp_path, monkeypatch,
                                                        directory, onefile, name) -> None:
    source = tmp_path / "mi agente"
    source.mkdir()
    entry = source / ("app.py" if directory else "agente.py")
    entry.write_text("raise AssertionError('Must not execute at build time')", encoding="utf-8")
    (source / "assets").mkdir()
    (source / "assets" / "ui.tcss").write_text("Screen {}", encoding="utf-8")
    (source / ".env").write_text("DO_NOT_BUNDLE=secret", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    expected_name = name or (source.name if directory else entry.stem)
    exe = expected_name if expected_name.endswith(".exe") else expected_name + ".exe"
    expected = tmp_path / "dist" / (exe if onefile else expected_name)

    def build(command, **kwargs):
        assert command[:3] == [sys.executable, "-m", "PyInstaller"]
        assert "--console" in command
        assert ("--onefile" if onefile else "--onedir") in command
        assert command[command.index("--name") + 1] == expected_name
        assert command[command.index("--paths") + 1] == str(source)
        assert command[command.index("--add-data") + 1] == f"{source / 'assets'};assets"
        assert command[command.index("--distpath") + 1] == str(tmp_path / "dist")
        assert "--additional-hooks-dir" not in command
        assert command[-1] == str(entry)
        assert not any(".env" in arg for arg in command)
        assert kwargs == {"cwd": source, "check": True}
        artifact = expected if onefile else expected / exe
        artifact.parent.mkdir(parents=True)
        artifact.touch()

    with patch("nas.adk.tui.build.sys.platform", "win32"), \
         patch("nas.adk.tui.build.find_spec", return_value=object()), \
         patch("nas.adk.tui.build.subprocess.run", side_effect=build):
        assert build_exe(source if directory else entry, name=name, onefile=onefile) == expected


@pytest.mark.parametrize("name", ["", "../bad", "a\\b", "a:b", "a\n", "bad.", "bad ",
                                  "CON", "nul.exe", "COM1", "LPT²", "CONIN$"])
def test_invalid_names(tmp_path, name) -> None:
    entry = tmp_path / "app.py"
    entry.touch()
    with pytest.raises(ValueError, match="valid Windows filename"):
        build_exe(entry, name=name)


def test_validation_and_build_failures(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="app.py"):
        build_exe(tmp_path)
    entry = tmp_path / "app.py"
    entry.touch()
    wrong = tmp_path / "app.txt"
    wrong.touch()
    with pytest.raises(ValueError, match=".py file"):
        build_exe(wrong)
    with patch("nas.adk.tui.build.sys.platform", "linux"):
        with pytest.raises(RuntimeError, match="built on Windows"):
            build_exe(entry)
    with patch("nas.adk.tui.build.sys.platform", "win32"), \
         patch("nas.adk.tui.build.find_spec", return_value=None):
        with pytest.raises(RuntimeError, match=r"nas-adk\[tui,build\]"):
            build_exe(entry)
    with patch("nas.adk.tui.build.sys.platform", "win32"), \
         patch("nas.adk.tui.build.find_spec", return_value=object()), \
         patch("nas.adk.tui.build.subprocess.run") as run:
        run.side_effect = subprocess.CalledProcessError(2, ["PyInstaller"])
        with pytest.raises(subprocess.CalledProcessError):
            build_exe(entry)
        run.side_effect = None
        with pytest.raises(RuntimeError, match="expected executable"):
            build_exe(entry)
        (tmp_path / "dist").mkdir()
        (tmp_path / "dist" / "app.exe").touch()
        run.reset_mock()
        with pytest.raises(FileExistsError, match="already exists"):
            build_exe(entry)
        run.assert_not_called()


def test_cli_defaults_options_and_errors() -> None:
    runner = CliRunner()
    with patch("nas.adk.tui.build.build_exe", return_value=Path("dist/MiAsistente.exe")) as build:
        result = runner.invoke(main, ["app.py", "--name", "MiAsistente"])
        assert result.exit_code == 0, result.output
        build.assert_called_once_with(Path("app.py"), name="MiAsistente", onefile=True)
        assert "Built:" in result.output
        runner.invoke(main, ["my_app", "--onedir"])
        build.assert_called_with(Path("my_app"), name=None, onefile=False)
        build.side_effect = subprocess.CalledProcessError(3, ["PyInstaller"])
        result = runner.invoke(main, ["app.py"])
        assert result.exit_code == 1 and "exit code 3" in result.output
        build.side_effect = FileNotFoundError("missing app.py")
        result = runner.invoke(main, ["my_app"])
        assert result.exit_code == 1 and "missing app.py" in result.output


@pytest.mark.skipif(sys.platform != "win32" or os.getenv("NAS_TUI_BUILD_SMOKE") != "1",
                    reason="Opt-in Windows EXE check: NAS_TUI_BUILD_SMOKE=1")
def test_real_onefile_offline_tui(tmp_path, monkeypatch) -> None:
    pytest.importorskip("PyInstaller")
    source = tmp_path / "offline agent"
    source.mkdir()
    (source / "assets").mkdir()
    (source / "assets" / "ui.tcss").write_text("#brand { color: $primary; }", encoding="utf-8")
    (source / "tools.py").write_text(
        "from nas.adk.models import BaseLlm, LlmResponse\n"
        "class OfflineLlm(BaseLlm):\n"
        "    model: str = 'offline'\n"
        "    async def generate_content_async(self, request):\n"
        "        return LlmResponse(content='Offline OK')\n", encoding="utf-8",
    )
    (source / "app.py").write_text(
        "import asyncio, sys\n"
        "from pathlib import Path\n"
        "from nas.adk import Agent\n"
        "from nas.adk.tui import AgentApp, ChatMessage, TuiConfig\n"
        "from tools import OfflineLlm\n"
        "ui = TuiConfig(css_path=Path(__file__).resolve().parent / 'assets' / 'ui.tcss')\n"
        "app = AgentApp(Agent(name='offline', model=OfflineLlm()), config=ui)\n"
        "async def check():\n"
        "    async with app.run_test():\n"
        "        app.composer.load_text('Hello')\n"
        "        await app.action_send()\n"
        "        await asyncio.wait_for(app.workers.wait_for_complete(), 10)\n"
        "        assert any(m.message_text == 'Offline OK' for m in app.chat.query(ChatMessage))\n"
        "    print('Frozen TUI OK')\n"
        "if __name__ == '__main__':\n"
        "    if '--smoke-test' in sys.argv:\n"
        "        asyncio.run(check())\n"
        "    else:\n"
        "        app.run()\n", encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    executable = build_exe(source, name="OfflineAgent")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    env = {key: value for key, value in os.environ.items()
           if key.upper() not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}}
    env["PATH"] = str(Path(os.environ["SystemRoot"]) / "System32")
    result = subprocess.run([str(executable), "--smoke-test"], cwd=elsewhere, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Frozen TUI OK" in result.stdout
    import pefile

    with pefile.PE(str(executable)) as pe:
        assert pe.OPTIONAL_HEADER.Subsystem == 3  # Windows console, not a windowed executable.
