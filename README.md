# Python project template

Authors:

Explain what the project is about and any other things users may need to know.

## Project setup and management with uv
https://docs.astral.sh/uv/

### Install uv
```bash
pipx install uv==0.9.5
```

### Sync (install) project dependencies
```bash
uv sync
```
If the project is version controlled, it is important to add the poetry.lock file to the project repository.

Add dependencies using `uv add packagename`

For compatibility with other tools, one can export the uv.lock to pylock.toml

## Ruff formatter / linter
https://docs.astral.sh/ruff/

Ruff is part of the `dev` dependency group, which is synced to the workspace by default.
It can be used in the command line
```bash
uv run ruff check   # Lint all files in the current directory.
uv run ruff format  # Format all files in the current directory.
```

In VSCode, the Ruff extension is sufficient, no other installs are necessary.