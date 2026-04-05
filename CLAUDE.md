## First Principles

1. Avoid including comments or docstrings unless absolutely necessary.
2. Never use emojis in your output code generation.
3. Do not generate `README.md` or `.md` files during code generation unless instructed to do so.
4. Avoid `__init__.py` files - use direct imports instead.

## Utilities

Aggregate targets (run from root directory):

- `make check` - Check all packages
- `make lint` - Lint all packages
- `make test` - Test all packages
- `make typecheck` - Type check all packages

## Miscellaneous

Shared configuration files:
- `ruff.toml` - Ruff linting/formatting (auto-discovered by all packages)
- `mypy.ini` - Mypy type checking configuration
- `pyrightconfig.json` - Pyright type checking configuration
- `poetry.toml` - Poetry virtualenv settings
