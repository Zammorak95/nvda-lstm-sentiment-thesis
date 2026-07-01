# Common workflows
.PHONY: install format lint test run scientific scientific-test

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
	python -m thesis.cli --help

scientific:
	python -m thesis.eval.make_scientific_outputs

scientific-test:
	python -m thesis.eval.make_scientific_outputs --run-pytest
