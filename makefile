SHELL := /bin/zsh

.PHONY: install lint check typecheck test fmt hooks dev

install:
	poetry install

fmt: lint

hooks:
	git config core.hooksPath .githooks

dev: hooks install

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
