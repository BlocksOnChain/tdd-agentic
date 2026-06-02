#!/usr/bin/env sh
# Thin wrapper for install_backend.py (macOS, Linux, Git Bash on Windows).
set -e
cd "$(dirname "$0")/.."
exec python3 scripts/install_backend.py "$@"
