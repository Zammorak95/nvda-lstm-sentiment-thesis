# Common workflows
.PHONY: install format lint test run

install:
	python -m pip install --upgrade pip
	pip install -e ".[dev]"

format:
	black .

lint:
	ruff check .

test:
	pytest

run:
	python -m thesis.cli greet --name "World"
