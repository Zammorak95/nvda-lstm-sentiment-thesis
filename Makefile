# Common workflows
.PHONY: help install format lint test run scientific scientific-test baselines baselines-ablations baselines-linear-svm baselines-linear-svm-ablations lstm-tune lstm-walkforward model-comparison reproduce-lite

help:
	@echo "Available targets:"
	@echo "  install                         Install project in editable mode"
	@echo "  scientific                      Generate dataset diagnostics and figures"
	@echo "  baselines-linear-svm            Run final classical baselines"
	@echo "  baselines-linear-svm-ablations  Run final baselines with feature ablations"
	@echo "  lstm-tune                       Run LSTM random search"
	@echo "  lstm-walkforward                Run final LSTM walk-forward specification"
	@echo "  model-comparison                Generate combined LSTM + baseline tables"
	@echo "  reproduce-lite                  Run non-LSTM reproduction outputs"
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

scientific:
	python -m thesis.eval.make_scientific_outputs

scientific-test:
	python -m thesis.eval.make_scientific_outputs --run-pytest

# Original baseline runner. Kept for compatibility; final thesis tables should use linear SVM.
baselines:
	python -m thesis.eval.run_baseline_models

# Original baseline ablations. Kept for compatibility; final thesis tables should use linear SVM.
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
