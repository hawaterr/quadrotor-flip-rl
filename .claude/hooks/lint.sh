#!/usr/bin/env bash
# PostToolUse hook: after Claude edits a .py file, run ruff on it and mypy on the project.
# Exit 2 sends the output back to Claude to fix; exit 0 means nothing to report.
file=$(jq -r '.tool_input.file_path // empty')
[[ "$file" == *.py ]] || exit 0
cd "$CLAUDE_PROJECT_DIR" || exit 0
[[ -x .venv/bin/ruff && -x .venv/bin/mypy ]] || exit 0  # dev tools not installed: pip install -r requirements-dev.txt

out=""
ruff_out=$(.venv/bin/ruff check --quiet "$file" 2>&1) || out+="$ruff_out"$'\n'
mypy_out=$(.venv/bin/mypy 2>&1) || out+="$mypy_out"$'\n'
if [[ -n "$out" ]]; then
    echo "$out" >&2
    exit 2
fi
