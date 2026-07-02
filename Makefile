# Common workflows
.PHONY: install format lint test run scientific scientific-test baselines baselines-ablations baselines-linear-svm baselines-linear-svm-ablations

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

baselines:
	python -m thesis.eval.run_baseline_models

baselines-ablations:
	python -m thesis.eval.run_baseline_models --run-ablations

baselines-linear-svm:
	python -m thesis.eval.run_baseline_models_linear_svm

baselines-linear-svm-ablations:
	python -m thesis.eval.run_baseline_models_linear_svm --run-ablations
