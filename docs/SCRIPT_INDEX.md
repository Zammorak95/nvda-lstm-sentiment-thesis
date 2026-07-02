# Script index

This file explains which scripts are canonical for reproduction and which files are retained for backwards compatibility.

## Canonical commands

| Purpose | Preferred command | Main output |
|---|---|---|
| StockData.org raw market/news downloads | `thesis-fetch-stockdata` | `data/raw/` |
| Full preprocessing pipeline | `thesis-preprocess all` | `data/model_feed/model_dataset_clean.csv` |
| Preprocessing validation/audit | `thesis-preprocess validate` / `thesis-preprocess audit` | `data/model_feed/model_dataset_audit.xlsx` |
| Dataset diagnostics and thesis figures | `python -m thesis.eval.make_scientific_outputs` | `artifacts/reports/scientific_outputs/` |
| Classical benchmarks with linear SVM | `python -m thesis.eval.run_baseline_models_linear_svm` | `artifacts/reports/baseline_models_linear_svm/` |
| Feature-set ablations | `python -m thesis.eval.run_baseline_models_linear_svm --run-ablations` | `artifacts/reports/baseline_models_linear_svm_ablations/` |
| LSTM random search | `thesis-tune-lstm` | `artifacts/models/random_search_direction_v2/` |
| LSTM walk-forward evaluation | `thesis-walkforward-lstm` | `artifacts/models/walk_forward_direction*/` |
| LSTM statistical report | `python src/thesis/eval/thesis_walkforward_report.py` | `artifacts/reports/lstm_walkforward*/` |
| Combined model comparison table | `thesis-model-comparison` | `artifacts/reports/model_comparison/` |

## Active source files

| File | Status | Notes |
|---|---|---|
| `src/thesis/data_acquisition/stockdata_api.py` | Active | Canonical StockData.org downloader for EOD/intraday market data and news headlines. |
| `src/thesis/preprocessing/data_pipeline.py` | Active | Canonical preprocessing pipeline: news, sentiment, EOD market features, Google Trends reconstruction, dataset merge, validation and audit. |
| `src/thesis/eval/make_scientific_outputs.py` | Active | Dataset tables, target distribution, feature correlations, thesis-ready diagnostic figures. |
| `src/thesis/eval/run_baseline_models_linear_svm.py` | Active | Preferred benchmark runner: majority class, logistic regression, Random Forest, linear SVM. |
| `src/thesis/eval/run_baseline_models.py` | Active but optional | Original benchmark runner. Includes `svm_rbf`; do not use for final thesis results unless explicitly discussed. |
| `src/thesis/eval/make_model_comparison_table.py` | Active | Combines LSTM and classical baselines into CSV/Markdown/LaTeX/PNG outputs. |
| `src/thesis/eval/thesis_walkforward_report.py` | Active | Statistical and trading-oriented report for LSTM walk-forward predictions. |
| `src/thesis/model_training/optimalisation/tune_lstm_direction.py` | Active wrapper | Canonical name for LSTM random-search tuning. Wraps the legacy script. |
| `src/thesis/model_training/optimalisation/evaluate_lstm_walk_forward.py` | Active wrapper | Canonical name for final LSTM walk-forward evaluation. Wraps the legacy script. |
| `src/thesis/model_training/optimalisation/random_search_lstm_direction_v2.py` | Legacy active | Original thesis random-search script. Kept for compatibility. |
| `src/thesis/model_training/optimalisation/walk_forward_lstm_direction_rocm.py` | Legacy active | Original thesis walk-forward script. Kept for compatibility and ROCm/GPU support. |

## Older data-acquisition/preprocessing files

The earlier loose acquisition and preprocessing scripts are now represented by canonical commands:

- `nvda_stockdata_fetch_combined.py` -> `thesis-fetch-stockdata`
- `data_pipeline_combined.py` -> `thesis-preprocess`
- `Stock_preprocessing.py` -> `thesis-preprocess stock-clean-all`
- `daily google trends series.py` -> `thesis-preprocess trends-reconstruct` and `thesis-preprocess trends-clean`
- `model_dataset_clean_and_analyze.py` -> `thesis-preprocess validate`, `thesis-preprocess audit`, and `python -m thesis.eval.make_scientific_outputs`

Do not use the old loose names as the primary reproduction instructions; use the canonical commands instead.

## Naming convention going forward

New scripts should follow this pattern:

```text
data_acquisition/*  Download raw external data.
preprocessing/*     Create/clean/merge datasets.
make_*              Generate tables/figures/reports without fitting a model.
run_*               Fit/evaluate benchmark models.
tune_*              Search hyperparameters.
evaluate_*          Final model evaluation.
```

Examples:

```text
data_acquisition/stockdata_api.py
preprocessing/data_pipeline.py
make_scientific_outputs.py
run_baseline_models_linear_svm.py
tune_lstm_direction.py
evaluate_lstm_walk_forward.py
make_model_comparison_table.py
```

## Files not recommended for final thesis reporting

- `run_baseline_models.py` with `svm_rbf` should not be used for the final thesis tables unless the RBF kernel is explicitly justified in the methodology.
- Archived files under `archive/` should be treated as historical references only.
- Local artifacts under `artifacts/` and `models/` should be regenerated or shared through a release package, not treated as source code.
