# Reproducibility guide

This guide describes how to reproduce the thesis workflow from a clean clone to the final tables and figures.

## Scope

The reproducible workflow has four levels:

1. **Report-level reproduction**: recreate tables and figures from `data/model_feed/model_dataset_clean.csv`.
2. **Model-level reproduction**: rerun baselines and LSTM walk-forward validation from the cleaned dataset.
3. **Preprocessing-level reproduction**: rebuild `model_dataset_clean.csv` from existing raw/intermediate files.
4. **Raw-data acquisition reproduction**: fetch market/news data through StockData.org and manually export Google Trends files.

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

## Step 2A — Acquire raw StockData.org data and Google Trends files

Set a local StockData.org token as `STOCKDATA_API_TOKEN` or `STOCKDATA_API_KEY`. Do not commit the token.

```cmd
thesis-fetch-stockdata market --mode eod --symbol NVDA --start 2019-03-01 --end 2026-03-01 --csv
thesis-fetch-stockdata market --mode eod --symbol SPY --start 2019-03-01 --end 2026-03-01 --csv --outdir data\raw\macro_stock_data\SPY
thesis-fetch-stockdata market --mode eod --symbol SOXX --start 2019-03-01 --end 2026-03-01 --csv --outdir data\raw\macro_stock_data\SOXX
thesis-fetch-stockdata market --mode eod --symbol IEF --start 2019-03-01 --end 2026-03-01 --csv --outdir data\raw\macro_stock_data\IEF
thesis-fetch-stockdata news --symbols NVDA --start 2019-03-01 --end 2026-03-01 --chunk-days 30 --csv
```

Google Trends files were downloaded manually from the Google Trends website: one full-period monthly overview plus shorter daily-window exports. Save those files in `data/raw/trends/`. See `docs/DATA_ACQUISITION.md` for the process.

## Step 2B — Reproduce from raw/intermediate data

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

Use these commands to inspect the acquisition and preprocessing subcommands:

```cmd
thesis-fetch-stockdata --help
thesis-preprocess --help
```

See `docs/DATA_ACQUISITION.md` and `docs/PREPROCESSING.md` for stage-by-stage explanations.

## Step 2C — Reproduce from the cleaned dataset

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

This evaluates majority class, logistic regression, linear SVM and Random Forest, including feature-set ablations.

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

On Linux/ROCm, add `--gpu 0`.

The historical thesis best run achieved an OOS AUC of approximately `0.5506`. A fresh retraining run may not match exactly because TensorFlow/LSTM training is stochastic.

## Step 6 — Generate LSTM statistical report

```cmd
python src\thesis\eval\thesis_walkforward_report.py ^
  --oos artifacts\models\walk_forward_direction_bestparams_reproduction\walk_forward_oos_predictions.csv ^
  --summary artifacts\models\walk_forward_direction_bestparams_reproduction\walk_forward_summary.json ^
  --outdir artifacts\reports\lstm_walkforward_bestparams_reproduction ^
  --cost_bps 5
```

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

- Raw-data reproduction requires a StockData.org token, the correct subscription access, manually exported Google Trends files and respect for data licences.
- Raw news and market data may not be legally redistributable.
- Google Trends exports can depend on query settings, geography, category, time window and export timing.
- Random Forest results are seeded and should be stable.
- Logistic regression and linear SVM should be stable given the same data and package versions.
- LSTM training can vary across repeated runs.
- GPU/CPU differences can change neural-network results.
- The random-search validation result should not be reported as the final result; the final assessment should use walk-forward out-of-sample predictions.
