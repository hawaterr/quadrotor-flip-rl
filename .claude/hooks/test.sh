#!/usr/bin/env bash
# Stop hook: when Claude ends a turn with uncommitted .py changes, run the tests.
# Exit 2 sends the failures back to Claude to fix; exit 0 lets the turn end.
[[ $(jq -r '.stop_hook_active // false') == true ]] && exit 0  # already continuing after a failure: don't loop
cd "$CLAUDE_PROJECT_DIR" || exit 0
[[ -x .venv/bin/pytest ]] || exit 0  # dev tools not installed: pip install -r requirements-dev.txt
[[ -n $(git status --porcelain -- '*.py') ]] || exit 0

if ! out=$(.venv/bin/pytest -x -q 2>&1); then
    echo "$out" | tail -n 40 >&2
    exit 2
fi
