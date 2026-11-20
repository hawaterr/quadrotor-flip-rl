- Ruff and mypy (config in `pyproject.toml`) check what they can; a hook runs them after every edit.
Dev tools: `pip install -r requirements-dev.txt`.
Edit `.py` files with the Edit/Write tools, not sed or scripts, so the lint hook runs.
- Do not commit unless instructed to
