# Script index

This file explains which scripts are canonical for reproduction and which files are retained for backwards compatibility.

## Canonical commands

| Purpose | Preferred command | Main output |
|---|---|---|
| Generic full stock pipeline | `bash scripts/run_stock_full_pipeline.sh main` | Data, LSTM report, baselines, model comparison, summary |
| Full pipeline including fixed comparisons | `bash scripts/run_stock_full_pipeline.sh all` | Same as main plus fixed comparison runs |
| Historical 0.5506-style LSTM check | `bash scripts/run_stock_full_pipeline.sh legacy_05506` | `artifacts/models/<symbol>_walk_forward_legacy_05506_params/` |
| Data stage only | `bash scripts/run_stock_full_pipeline.sh data` | `data/model_feed/<symbol>_model_dataset.csv` |
| Thesis clean dataset stage | `bash scripts/run_stock_full_pipeline.sh reduced` | `data/model_feed/<symbol>_model_dataset_clean.csv` |
| Random search only | `bash scripts/run_stock_full_pipeline.sh random_search` | `artifacts/models/<symbol>_random_search_reduced_features/` |
| Best-params walk-forward only | `bash scripts/run_stock_full_pipeline.sh walk_bestparams` | `artifacts/models/<symbol>_walk_forward_random_search_bestparams/` |
| Classical baselines and ablations | `bash scripts/run_stock_full_pipeline.sh baselines` | `artifacts/reports/<symbol>_baseline_models_linear_svm_ablations/` |
| Combined model comparison | `bash scripts/run_stock_full_pipeline.sh model_comparison` | `artifacts/reports/<symbol>_model_comparison/` |

## Active source files

| File | Status | Notes |
|---|---|---|
| `scripts/run_stock_full_pipeline.sh` | Active | Main generic end-to-end pipeline for NVDA, AMD or another ticker. |
| `src/thesis/pipelines/run_stock_pipeline.py` | Active | Generic stock data pipeline: EOD, news, PyTrends, sentiment, model dataset, audit. |
| `src/thesis/pipelines/run_stock_pipeline_auto_window.py` | Active | Auto-discovers earliest news date with an API-conscious hierarchical scan. |
| `src/thesis/One doc/nvda_stockdata_fetch_combined.py` | Legacy dependency | Still used by the generic pipeline for StockData EOD/intraday fetching. |
| `src/thesis/One doc/data_pipeline_combined.py` | Legacy dependency | Still used by the generic pipeline for preprocessing, merge and audit operations. |
| `src/thesis/eval/run_baseline_models_linear_svm.py` | Active | Preferred benchmark runner: majority class, logistic regression, Random Forest, linear SVM. |
| `src/thesis/eval/make_model_comparison_table.py` | Active | Combines LSTM and classical baselines into CSV/Markdown/LaTeX/PNG outputs. |
| `src/thesis/eval/thesis_walkforward_report.py` | Active | Statistical and trading-oriented report for LSTM walk-forward predictions. |
| `src/thesis/model_training/optimalisation/random_search_lstm_direction_v2.py` | Active | LSTM random search; includes the 96-unit option needed for the historical best-parameter route. |
| `src/thesis/model_training/optimalisation/walk_forward_lstm_direction_rocm.py` | Active | Walk-forward LSTM evaluation with ROCm/GPU support. |

## Main route

```text
data -> reduced -> random_search -> walk_bestparams -> report_bestparams -> baselines -> model_comparison -> summary
```

## Historical NVDA result route

The old thesis result around OOS AUC `0.5506` is not hard-coded into the main pipeline. Instead, the old parameter setting is available as a reproducible optional route:

```bash
SYMBOL=NVDA KEYWORD="NVIDIA stock" END=2026-03-01 GPU=0 \
  bash scripts/run_stock_full_pipeline.sh legacy_05506
```

This keeps the pipeline generic while preserving a way to rerun the historical best-parameter specification.

## Naming convention going forward

New scripts should follow this pattern:

```text
pipelines/*          Orchestrate raw data, preprocessing and modelling workflows.
data_acquisition/*   Download raw external data.
preprocessing/*      Create/clean/merge datasets.
make_*               Generate tables/figures/reports without fitting a model.
run_*                Fit/evaluate benchmark models.
tune_*               Search hyperparameters.
evaluate_*           Final model evaluation.
```

## Files not recommended for final thesis reporting

- Archived files under `archive/` should be treated as historical references only.
- Local artifacts under `artifacts/` and `models/` should be regenerated or shared through the pipeline, not edited manually.
