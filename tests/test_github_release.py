from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


def test_release_workflow_publishes_github_release() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load(
        (root / ".github/workflows/release.yml").read_text(encoding="utf-8")
    )
    text = (root / ".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "contents: write" in text
    assert "softprops/action-gh-release@v2" in text
    assert "actions/upload-artifact@v4" in text
    assert "azure-pipelines.yml" not in text
    assert "release" in workflow["jobs"]


def test_release_bump_and_skip(tmp_path: Path) -> None:
    pytest.importorskip("commitizen")
    tomlkit = pytest.importorskip("tomlkit")
    git = shutil.which("git")
    assert git is not None
    bash = shutil.which("bash")
    assert bash is not None
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/ci/bump_version.sh").read_text(encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    remote = tmp_path / "remote.git"
    output = tmp_path / "github-output"
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GITHUB_OUTPUT": str(output),
        "PYTHON": sys.executable,
    }
    # GitHub Actions exporta el SHA del repositorio real. El test usa un repo
    # temporal, así que debe dejar que el script resuelva HEAD localmente.
    env.pop("GITHUB_SHA", None)

    def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args, cwd=repo, env=env, check=check, capture_output=True, text=True, timeout=60
        )

    def bump() -> subprocess.CompletedProcess[str]:
        output.write_text("", encoding="utf-8")
        return run(bash, "-c", script, check=False)

    run(git, "init", "--bare", str(remote))
    run(git, "init", "-b", "main")
    run(git, "config", "user.name", "Release test")
    run(git, "config", "user.email", "release@example.invalid")
    config = tomlkit.parse((root / "pyproject.toml").read_text(encoding="utf-8"))
    config["tool"]["commitizen"]["version"] = "0.1.0"
    config["tool"]["commitizen"]["version_files"] = ["nas/adk/version.py:__version__"]
    (repo / "pyproject.toml").write_text(tomlkit.dumps(config), encoding="utf-8")
    version = repo / "nas/adk/version.py"
    version.parent.mkdir(parents=True)
    version.write_text('__version__ = "0.1.0"\n')
    run(git, "add", ".")
    run(git, "commit", "-m", "chore: initialize release")
    run(git, "remote", "add", "origin", str(remote))
    run(git, "push", "origin", "main")

    result = bump()
    assert result.returncode == 0, result.stderr
    assert "release=false" in output.read_text(encoding="utf-8")

    run(git, "commit", "--allow-empty", "-m", "feat: add a feature")
    run(git, "push", "origin", "main")
    result = bump()
    assert result.returncode == 0, result.stderr
    lines = output.read_text(encoding="utf-8")
    assert "release=true" in lines and "version=0.2.0" in lines
    assert '__version__ = "0.2.0"' in version.read_text(encoding="utf-8")
    assert "[skip ci]" in run(git, "log", "-1", "--format=%s").stdout
    assert "v0.2.0" in run(git, "ls-remote", "--tags", "origin").stdout
    assert (repo / "CHANGELOG.md").is_file()

    result = bump()
    assert result.returncode == 0, result.stderr
    assert "release=false" in output.read_text(encoding="utf-8")

    version.write_text('__version__ = "9.9.9"\n')
    run(git, "add", ".")
    run(git, "commit", "-m", "fix: inconsistent version")
    run(git, "push", "origin", "main")
    result = bump()
    assert result.returncode != 0
    assert "release=true" not in output.read_text(encoding="utf-8")
