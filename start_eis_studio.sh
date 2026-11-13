#!/usr/bin/env bash
# EIS Studio - start on macOS or Linux.
#   bash start_eis_studio.sh            set up (first time) and start
#   bash start_eis_studio.sh --bayes    also install CmdStan for Bayesian DRT
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for c in python3.12 python3.11 python3.13 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && \
     "$c" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done

if [ -z "$PY" ]; then
  echo
  echo "  EIS Studio needs Python 3.10 - 3.13, which was not found."
  if [ "$(uname)" = "Darwin" ]; then
    echo "  Install it from https://www.python.org/downloads/  (or: brew install python@3.12)"
    open "https://www.python.org/downloads/" 2>/dev/null || true
  else
    echo "  Install it with your package manager, e.g.:"
    echo "    Ubuntu/Debian:  sudo apt install python3 python3-venv"
    echo "    Fedora:         sudo dnf install python3"
  fi
  echo
  exit 1
fi

exec "$PY" launcher.py "$@"
