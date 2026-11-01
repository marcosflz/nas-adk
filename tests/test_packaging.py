from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path


def test_wheel_contents_and_metadata_without_legacy_sdk(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", str(tmp_path)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    wheel = next(tmp_path.glob("*.whl"))
    extracted = tmp_path / "wheel"
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "nas/adk/__init__.py" in names
        assert not any(name.startswith("edhm/") for name in names)
        assert "nas/adk/models/opencode.py" in names
        assert "nas/adk/models/openrouter.py" in names
        assert "nas/adk/models/typesafe.py" in names
        assert "nas/adk/tui/__init__.py" in names
        metadata = next(name for name in names if name.endswith("/METADATA"))
        text = archive.read(metadata).decode()
        assert "Requires-Dist: edhm-sdk" not in text
        assert "Requires-Dist: httpx" in text
        assert "Requires-Dist: pydantic" in text
        assert "Requires-Dist: rich" in text
        archive.extractall(extracted)

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import sys, inspect; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
            "from nas.adk import Agent, Runner; "
            "from nas.adk.models import OpenCodeLlm, OpenRouterLlm, TypesafeLlm; "
            "assert Path(inspect.getfile(Agent)).is_relative_to(Path(sys.argv[1]))",
            str(extracted),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
