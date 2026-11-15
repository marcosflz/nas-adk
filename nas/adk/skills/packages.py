"""Bounded imports and exports; importing never runs code or installs packages."""

from __future__ import annotations

import base64
import io
import hashlib
import json
import re
import stat
import zipfile
from pathlib import Path
from typing import Awaitable, Callable
from urllib.parse import quote, unquote, urlparse

import yaml

from .models import MAX_FILES, MAX_PACKAGE_BYTES, SkillManifest, SkillPackage
from .models import package_digest, safe_skill_path

SourceFetcher = Callable[[str, int], Awaitable[bytes]]


def parse_skill_md(text: str) -> tuple[SkillManifest, str]:
    text = text.removeprefix("\ufeff").replace("\r\n", "\n")
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise ValueError("SKILL.md must start with YAML frontmatter")
    try:
        end = lines.index("---", 1)
    except ValueError as error:
        raise ValueError("Unclosed SKILL.md frontmatter") from error
    try:
        data = yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError as error:
        raise ValueError("Invalid YAML frontmatter in SKILL.md") from error
    if not isinstance(data, dict):
        raise ValueError("SKILL.md frontmatter must be a mapping")
    known = {key: value for key, value in data.items()
             if key in {"name", "description", "license", "compatibility", "metadata"}}
    if "allowed-tools" in data:
        known["allowedTools"] = data["allowed-tools"]
    manifest = SkillManifest.model_validate(known)
    body = "\n".join(lines[end + 1:]).strip()
    if not body:
        raise ValueError("SKILL.md must contain instructions")
    return manifest, body


def package_from_files(files: dict[str, bytes]) -> SkillPackage:
    if "SKILL.md" not in files:
        raise ValueError("Missing SKILL.md")
    if len(files) > MAX_FILES or sum(map(len, files.values())) > MAX_PACKAGE_BYTES:
        raise ValueError("Skill package exceeds its file or size limit")
    manifest, body = parse_skill_md(files["SKILL.md"].decode("utf-8"))
    encoded = {safe_skill_path(path): base64.b64encode(data).decode()
               for path, data in files.items()}
    return SkillPackage(manifest=manifest, instructions=body, files=encoded,
                        digest=package_digest(encoded))


def import_directory(root: str | Path) -> SkillPackage:
    if Path(root).is_symlink():
        raise ValueError("Symbolic links are not supported in skills")
    root = Path(root).resolve(strict=True)
    files: dict[str, bytes] = {}
    size = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symbolic links are not supported in skills")
        if path.is_file():
            if path.stat().st_size + size > MAX_PACKAGE_BYTES or len(files) >= MAX_FILES:
                raise ValueError("Skill package exceeds its file or size limit")
            data = path.read_bytes()
            size += len(data)
            files[path.relative_to(root).as_posix()] = data
    package = package_from_files(files)
    if root.name != package.manifest.name:
        raise ValueError("Skill directory must match its manifest name")
    return package


def import_zip(data: bytes, subdirectory: str = "") -> list[SkillPackage]:
    if len(data) > MAX_PACKAGE_BYTES:
        raise ValueError("Archive exceeds 10 MB")
    files: dict[str, bytes] = {}
    size = 0
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        if len(entries) > 4096:
            raise ValueError("Archive contains too many entries")
        for entry in entries:
            path = entry.filename.rstrip("/")
            safe_skill_path(path)
            mode = entry.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError("Symbolic links are not supported in skills")
            if entry.is_dir():
                continue
            if path in files or entry.flag_bits & 1:
                raise ValueError("Duplicate paths and encrypted archives are not supported")
            size += entry.file_size
            if size > MAX_PACKAGE_BYTES:
                raise ValueError("Expanded archive exceeds 10 MB")
            files[path] = archive.read(entry)
    return _packages_from_files(files, subdirectory)


def _packages_from_files(files: dict[str, bytes], subdirectory: str = "") -> list[SkillPackage]:
    roots = sorted(path[:-len("SKILL.md")] for path in files
                   if path == "SKILL.md" or path.endswith("/SKILL.md"))
    if subdirectory:
        suffix = safe_skill_path(subdirectory).rstrip("/") + "/"
        roots = [root for root in roots if root == suffix or root.endswith("/" + suffix)]
    if not roots:
        raise ValueError("No SKILL.md found in the archive")
    if len(roots) > 100:
        raise ValueError("Archive contains more than 100 skills")
    packages = []
    for root in roots:
        selected = {path[len(root):]: content for path, content in files.items()
                    if path.startswith(root) and not any(
                        other != root and other.startswith(root) and path.startswith(other)
                        for other in roots)}
        package = package_from_files(selected)
        if root and root.rstrip("/").split("/")[-1] != package.manifest.name:
            raise ValueError("Skill directory must match its manifest name")
        packages.append(package)
    return packages


def export_zip(package: SkillPackage) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package.files):
            archive.writestr(f"{package.manifest.name}/{path}", package.read(path))
    return output.getvalue()


async def import_github(url: str, fetch: SourceFetcher) -> list[SkillPackage]:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "github.com" or parsed.port:
        raise ValueError("Use an https://github.com repository URL")
    parts = [unquote(part) for part in parsed.path.strip("/").split("/")]
    if len(parts) < 2 or any(not part or part in {".", ".."} for part in parts):
        raise ValueError("Invalid GitHub repository URL")
    owner, repo = parts[:2]
    if not all(re.fullmatch(r"[A-Za-z0-9_.-]+", value) for value in (owner, repo)):
        raise ValueError("Invalid GitHub owner or repository name")
    ref, directory = "HEAD", ""
    if len(parts) > 2:
        if len(parts) < 4 or parts[2] != "tree":
            raise ValueError("Use a repository URL or /tree/ref/path URL")
        ref, directory = parts[3], "/".join(parts[4:])
    if directory:
        return await _import_github_directory(owner, repo.removesuffix(".git"), ref, directory, fetch)
    target = f"https://codeload.github.com/{quote(owner, safe='')}/{quote(repo.removesuffix('.git'), safe='')}/zip/{quote(ref, safe='')}"
    return import_zip(await fetch(target, MAX_PACKAGE_BYTES), directory)


async def _import_github_directory(owner: str, repo: str, ref: str, directory: str,
                                   fetch: SourceFetcher) -> list[SkillPackage]:
    """Fetch only selected skill blobs, even when the repository archive is very large."""
    prefix = safe_skill_path(directory).rstrip("/") + "/"
    base = f"https://api.github.com/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/git"
    tree = json.loads(await fetch(f"{base}/trees/{quote(ref, safe='')}?recursive=1", MAX_PACKAGE_BYTES))
    if not isinstance(tree, dict) or tree.get("truncated") or not isinstance(tree.get("tree"), list):
        raise ValueError("GitHub returned an incomplete tree; import a ZIP of the skill instead")
    entries = [entry for entry in tree["tree"] if isinstance(entry, dict)
               and str(entry.get("path", "")).startswith(prefix)]
    roots = [entry["path"][:-len("SKILL.md")] for entry in entries
             if str(entry.get("path", "")).endswith("/SKILL.md")]
    if not roots or len(roots) > 100:
        raise ValueError("The selected directory must contain between 1 and 100 skills")
    selected = [entry for entry in entries if any(entry["path"].startswith(root) for root in roots)
                and entry.get("type") != "tree"]
    if len(selected) > 4096 or any(entry.get("mode") not in {"100644", "100755"} for entry in selected):
        raise ValueError("Unsupported links, submodules or too many files in the selected skills")
    size = sum(int(entry.get("size", 0)) for entry in selected)
    if size > MAX_PACKAGE_BYTES:
        raise ValueError("Selected skills exceed 10 MB")
    files: dict[str, bytes] = {}
    for entry in selected:
        path = safe_skill_path(str(entry["path"])[len(prefix):])
        sha, expected = str(entry.get("sha", "")), int(entry.get("size", -1))
        if not re.fullmatch(r"[a-f0-9]{40}", sha) or not 0 <= expected <= MAX_PACKAGE_BYTES:
            raise ValueError("Invalid GitHub blob reference")
        blob = json.loads(await fetch(f"{base}/blobs/{sha}", int(expected * 1.5) + 4096))
        if not isinstance(blob, dict) or blob.get("encoding") != "base64":
            raise ValueError("GitHub returned an unsupported blob encoding")
        content = base64.b64decode("".join(str(blob.get("content", "")).split()), validate=True)
        header = f"blob {len(content)}\0".encode()
        if len(content) != expected or hashlib.sha1(header + content).hexdigest() != sha:
            raise ValueError("GitHub blob content does not match the selected version")
        files[path] = content
    packages = _packages_from_files(files)
    if "SKILL.md" in files and packages[0].manifest.name != prefix.rstrip("/").split("/")[-1]:
        raise ValueError("Skill directory must match its manifest name")
    return packages
