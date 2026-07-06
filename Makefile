# Common workflows
.PHONY: help install format lint test run pipeline-env pipeline-main pipeline-all pipeline-legacy fetch-nvda fetch-macro fetch-news preprocess preprocess-validate preprocess-audit scientific scientific-test baselines baselines-ablations baselines-linear-svm baselines-linear-svm-ablations lstm-tune lstm-walkforward model-comparison reproduce-lite

help:
	@echo "Available targets:"
	@echo "  install                         Install project in editable mode"
	@echo "  pipeline-env                    Check generic full-pipeline configuration"
	@echo "  pipeline-main                   Run generic full thesis pipeline without fixed comparisons"
	@echo "  pipeline-all                    Run generic full thesis pipeline including fixed comparisons"
	@echo "  pipeline-legacy                 Run historical 0.5506-parameter LSTM check"
	@echo "  fetch-nvda / fetch-macro/news   Legacy direct StockData fetch helpers"
	@echo "  preprocess                      Legacy preprocessing helper"
	@echo "  scientific                      Generate dataset diagnostics and figures"
	@echo "  baselines-linear-svm-ablations  Run final baselines with feature ablations"
	@echo "  model-comparison                Generate combined LSTM + baseline tables"
	@echo "  test / lint / format            Developer utilities"

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

pipeline-env:
	ROOT=$$(pwd) PYTHON=python bash scripts/run_stock_full_pipeline.sh env

pipeline-main:
	SYMBOL=NVDA KEYWORD="NVIDIA stock" END=2026-03-01 TRIALS=50 GPU=0 bash scripts/run_stock_full_pipeline.sh main

pipeline-all:
	SYMBOL=NVDA KEYWORD="NVIDIA stock" END=2026-03-01 TRIALS=50 GPU=0 bash scripts/run_stock_full_pipeline.sh all

pipeline-legacy:
	SYMBOL=NVDA KEYWORD="NVIDIA stock" END=2026-03-01 GPU=0 bash scripts/run_stock_full_pipeline.sh legacy_05506

fetch-nvda:
	thesis-fetch-stockdata market --mode eod --symbol NVDA --start 2019-03-01 --end 2026-03-01 --csv

fetch-macro:
	thesis-fetch-stockdata market --mode eod --symbol SPY --start 2019-03-01 --end 2026-03-01 --csv --outdir data/raw/macro_stock_data/SPY
	thesis-fetch-stockdata market --mode eod --symbol SOXX --start 2019-03-01 --end 2026-03-01 --csv --outdir data/raw/macro_stock_data/SOXX
	thesis-fetch-stockdata market --mode eod --symbol IEF --start 2019-03-01 --end 2026-03-01 --csv --outdir data/raw/macro_stock_data/IEF

fetch-news:
	thesis-fetch-stockdata news --symbols NVDA --start 2019-03-01 --end 2026-03-01 --chunk-days 30 --csv

preprocess:
	thesis-preprocess all

preprocess-validate:
	thesis-preprocess validate --input data/model_feed/model_dataset_clean.csv

preprocess-audit:
	thesis-preprocess audit --input data/model_feed/model_dataset_clean.csv --output data/model_feed/model_dataset_audit.xlsx

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
	python -m thesis.eval.run_baseline_models_linear_svm --run-ablations --outdir artifacts/reports/baseline_models_linear_svm_ablations

lstm-tune:
	thesis-tune-lstm --trials 50 --auto_threshold

lstm-walkforward:
	thesis-walkforward-lstm --data data/model_feed/model_dataset_clean.csv --outdir artifacts/models/walk_forward_direction_bestparams_reproduction --lookback 90 --initial_train 700 --val_size 126 --test_horizon 63 --step 63 --epochs 30 --batch 64 --lr 0.0003 --lstm_units 96 --dense_units 64 --dropout 0.10 --recurrent_dropout 0.20 --auto_threshold

model-comparison:
	thesis-model-comparison --baseline-metrics artifacts/reports/baseline_models_linear_svm_ablations/tables/baseline_model_metrics.csv --lstm-auc 0.550643920654932 --lstm-accuracy 0.5178571428571429 --lstm-sharpe 0.9957887190041333 --lstm-trade-rate 0.5396825396825397 --outdir artifacts/reports/model_comparison

reproduce-lite: scientific baselines-linear-svm-ablations model-comparison
