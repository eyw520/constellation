SHELL := /bin/zsh

.PHONY: install lint check typecheck test

install:
	poetry install

lint:
	poetry run ruff check src/ --unsafe-fixes --fix
	poetry run ruff format src/

check:
	poetry run ruff check src/
	poetry run ruff format --check src/

typecheck:
	poetry run mypy src/

test:
	poetry run pytest tests/ -v
