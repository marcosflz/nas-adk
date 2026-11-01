from __future__ import annotations

from pathlib import Path

import pytest

from nas.adk.cli.loader import load_agent


def test_load_agent_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_agent("/nonexistent/path/agent.py")


def test_load_agent_no_root_agent(tmp_path: Path) -> None:
    agent_dir = tmp_path / "my_agent"
    agent_dir.mkdir()
    agent_file = agent_dir / "agent.py"
    agent_file.write_text("x = 1")
    with pytest.raises(ValueError, match="root_agent"):
        load_agent(str(agent_dir))
