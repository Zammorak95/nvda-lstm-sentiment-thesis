# Reproducibility guide

This guide describes how to reproduce the thesis workflow from a clean clone to the final tables and figures.

## Scope

The reproducible workflow has three levels:

1. **Report-level reproduction**: recreate tables and figures from `data/model_feed/model_dataset_clean.csv`.
2. **Model-level reproduction**: rerun baselines and LSTM walk-forward validation from the cleaned dataset.
3. **Full raw-data reproduction**: rerun preprocessing from the original raw source files into `model_dataset_clean.csv`.

The most reliable and practical route is level 1 or 2. Exact retraining of the LSTM can vary slightly because neural-network training is stochastic and can differ across CPU/GPU, TensorFlow version and operating system.

## Expected repository state

Use branch:

```cmd
git switch main_v2
```

Expected clean input for report/model reproduction:

```text
data/model_feed/model_dataset_clean.csv
```

Expected generated outputs:

```text
artifacts/reports/scientific_outputs/
artifacts/reports/baseline_models_linear_svm_ablations/
artifacts/models/walk_forward_direction_bestparams_reproduction/
artifacts/reports/model_comparison/
```

## Step 1 — Create environment

Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python -m pip install tensorflow statsmodels tabulate
```

For non-LSTM report reproduction, TensorFlow is not required. It is only needed for LSTM random search and walk-forward training.

## Step 2A — Reproduce from raw/intermediate data

If the raw files are available in the expected layout, run:

```cmd
thesis-preprocess all
```

This creates:

```text
data/interim/news_headlines_master.csv
data/processed/news_headlines_clean.csv
data/processed/news_daily_sentiment.csv
data/processed/NVDA_eod_processed.csv
data/processed/SPY_eod_processed.csv
data/processed/SOXX_eod_processed.csv
data/processed/IEF_eod_processed.csv
data/interim/nvidia_trends_daily_consistent.csv
data/processed/nvidia_trends_processed.csv
data/model_feed/model_dataset.csv
data/model_feed/model_dataset_clean.csv
data/model_feed/model_dataset_audit.xlsx
```

Use this command to inspect preprocessing subcommands:

```cmd
thesis-preprocess --help
```

See `docs/PREPROCESSING.md` for a stage-by-stage explanation.

## Step 2B — Reproduce from the cleaned dataset

If raw files are unavailable, place the cleaned dataset here:

```text
data/model_feed/model_dataset_clean.csv
```

The modelling and reporting scripts use this path by default.

## Step 3 — Generate dataset diagnostics

```cmd
python -m thesis.eval.make_scientific_outputs
```

This creates dataset overview tables, class balance, feature descriptives, correlations, target distribution and thesis-ready diagnostic figures.

## Step 4 — Run classical baseline models

Use the linear-SVM version for the final thesis benchmark set:

```cmd
python -m thesis.eval.run_baseline_models_linear_svm ^
  --run-ablations ^
  --outdir artifacts\reports\baseline_models_linear_svm_ablations
```

This evaluates:

- Majority-class benchmark
- Logistic regression
- Linear SVM
- Random Forest

It also generates feature-set ablations for market, macro, sentiment and attention features.

## Step 5 — Reproduce the final LSTM walk-forward specification

```cmd
thesis-walkforward-lstm ^
  --data data\model_feed\model_dataset_clean.csv ^
  --outdir artifacts\models\walk_forward_direction_bestparams_reproduction ^
  --lookback 90 ^
  --initial_train 700 ^
  --val_size 126 ^
  --test_horizon 63 ^
  --step 63 ^
  --epochs 30 ^
  --batch 64 ^
  --lr 0.0003 ^
  --lstm_units 96 ^
  --dense_units 64 ^
  --dropout 0.10 ^
  --recurrent_dropout 0.20 ^
  --auto_threshold
```

On Linux/ROCm, add:

```bash
--gpu 0
```

The historical thesis best run achieved an OOS AUC of approximately `0.5506`. A fresh retraining run may not match exactly because TensorFlow/LSTM training is stochastic.

## Step 6 — Generate LSTM statistical report

```cmd
python src\thesis\eval\thesis_walkforward_report.py ^
  --oos artifacts\models\walk_forward_direction_bestparams_reproduction\walk_forward_oos_predictions.csv ^
  --summary artifacts\models\walk_forward_direction_bestparams_reproduction\walk_forward_summary.json ^
  --outdir artifacts\reports\lstm_walkforward_bestparams_reproduction ^
  --cost_bps 5
```

This creates LSTM-oriented figures and statistical checks, including calibration, equity/drawdown and fold-level metrics.

## Step 7 — Generate combined comparison table

Use the historical final LSTM values when reproducing the exact table from the thesis:

```cmd
thesis-model-comparison ^
  --baseline-metrics artifacts\reports\baseline_models_linear_svm_ablations\tables\baseline_model_metrics.csv ^
  --lstm-auc 0.550643920654932 ^
  --lstm-accuracy 0.5178571428571429 ^
  --lstm-sharpe 0.9957887190041333 ^
  --lstm-trade-rate 0.5396825396825397 ^
  --outdir artifacts\reports\model_comparison
```

If you want to use a fresh LSTM reproduction instead, pass the metrics from the new summary file or use:

```cmd
thesis-model-comparison ^
  --baseline-metrics artifacts\reports\baseline_models_linear_svm_ablations\tables\baseline_model_metrics.csv ^
  --lstm-summary artifacts\models\walk_forward_direction_bestparams_reproduction\walk_forward_summary.json ^
  --outdir artifacts\reports\model_comparison_reproduced
```

## Final output files to inspect

```text
data/model_feed/model_dataset_clean.csv
data/model_feed/model_dataset_audit.xlsx
artifacts/reports/scientific_outputs/
artifacts/reports/baseline_models_linear_svm_ablations/figures/
artifacts/reports/baseline_models_linear_svm_ablations/tables/
artifacts/reports/lstm_walkforward_bestparams_reproduction/
artifacts/reports/model_comparison/
```

## Reproducibility caveats

- Raw-data reproduction requires access to the original raw CSV/API exports and their licences.
- Random Forest results are seeded and should be stable.
- Logistic regression and linear SVM should be stable given the same data and package versions.
- LSTM training can vary across repeated runs.
- GPU/CPU differences can change neural-network results.
- The random-search validation result should not be reported as the final result; the final assessment should use walk-forward out-of-sample predictions.
