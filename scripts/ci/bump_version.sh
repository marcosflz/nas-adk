#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
OUTPUT="${GITHUB_OUTPUT:-/dev/stdout}"

git config user.name "GitHub Actions"
git config user.email "actions@github.com"
git fetch origin main --tags
git checkout -B main "${GITHUB_SHA:-HEAD}"
git merge --ff-only origin/main

set +e
"$PYTHON" -m commitizen bump --dry-run --yes --check-consistency
status=$?
set -e

if [ "$status" -eq 3 ] || [ "$status" -eq 21 ]; then
    echo "No eligible commits for bump. Skipping publication."
    echo "release=false" >> "$OUTPUT"
    exit 0
fi
if [ "$status" -ne 0 ]; then
    exit "$status"
fi

"$PYTHON" -m commitizen bump --yes --check-consistency
version=$("$PYTHON" - <<'PY'
import re
from pathlib import Path

text = Path("nas/adk/version.py").read_text(encoding="utf-8")
match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
if match is None:
    raise SystemExit("version.py does not define __version__")
print(match.group(1))
PY
)
git push origin HEAD:refs/heads/main --tags
echo "release=true" >> "$OUTPUT"
echo "version=$version" >> "$OUTPUT"
echo "Released $version"
