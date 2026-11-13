#!/usr/bin/env bash
# macOS: double-click to start EIS Studio (opens in Terminal).
# If macOS says the file cannot be opened: right-click it > Open, or run once in
# Terminal:  chmod +x "Start EIS Studio.command"
cd "$(dirname "$0")"
exec bash start_eis_studio.sh "$@"
