# Command reference

This document lists the canonical commands for reproducing the thesis workflow. Run commands from the repository root.

## 1. Environment

### Windows CMD

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

Install TensorFlow only when reproducing the LSTM runs:

```cmd
python -m pip install tensorflow statsmodels tabulate
```

## 2. Raw-data acquisition

Set your StockData.org token locally as `STOCKDATA_API_TOKEN` or `STOCKDATA_API_KEY`. Do not commit tokens.

Show available acquisition commands:

```cmd
thesis-fetch-stockdata --help
```

Download market/news data:

```cmd
thesis-fetch-stockdata market --mode eod --symbol NVDA --start 2019-03-01 --end 2026-03-01 --csv
thesis-fetch-stockdata market --mode eod --symbol SPY  --start 2019-03-01 --end 2026-03-01 --csv --outdir data\raw\macro_stock_data\SPY
thesis-fetch-stockdata market --mode eod --symbol SOXX --start 2019-03-01 --end 2026-03-01 --csv --outdir data\raw\macro_stock_data\SOXX
thesis-fetch-stockdata market --mode eod --symbol IEF  --start 2019-03-01 --end 2026-03-01 --csv --outdir data\raw\macro_stock_data\IEF
thesis-fetch-stockdata news --symbols NVDA --start 2019-03-01 --end 2026-03-01 --chunk-days 30 --csv
```

Google Trends is downloaded manually from the Google Trends website. Save one full-period monthly export plus smaller daily-window exports in:

```text
data/raw/trends/
```

See `docs/DATA_ACQUISITION.md` for the full process.

## 3. Raw/intermediate data location

Default raw-data locations:

```text
data/raw/news_headlines/
data/raw/stock_data/NVDA/
data/raw/macro_stock_data/SPY/
data/raw/macro_stock_data/SOXX/
data/raw/macro_stock_data/IEF/
data/raw/trends/
```

Expected cleaned model dataset:

```text
data/model_feed/model_dataset_clean.csv
```

The dataset is intentionally ignored by Git by default. See `docs/DATA.md` for guidance on whether to publish it.

## 4. Preprocessing pipeline

Show available preprocessing commands:

```cmd
thesis-preprocess --help
```

Full preprocessing run, assuming raw files are in the default locations:

```cmd
thesis-preprocess all
```

Individual stages:

```cmd
thesis-preprocess news-combine
thesis-preprocess news-clean
thesis-preprocess news-sentiment
thesis-preprocess stock-clean-all
thesis-preprocess trends-reconstruct
thesis-preprocess trends-clean
thesis-preprocess build-model
thesis-preprocess write-clean
thesis-preprocess validate --input data\model_feed\model_dataset_clean.csv
thesis-preprocess audit --input data\model_feed\model_dataset_clean.csv --output data\model_feed\model_dataset_audit.xlsx
```

## 5. Dataset diagnostics and thesis figures

```cmd
python -m thesis.eval.make_scientific_outputs
```

## 6. Classical baselines with linear SVM

```cmd
python -m thesis.eval.run_baseline_models_linear_svm ^
  --run-ablations ^
  --outdir artifacts\reports\baseline_models_linear_svm_ablations
```

## 7. LSTM hyperparameter tuning

```cmd
thesis-tune-lstm --trials 50 --auto_threshold
```

## 8. Final LSTM walk-forward reproduction

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

## 9. LSTM statistical report

```cmd
python src\thesis\eval\thesis_walkforward_report.py ^
  --oos artifacts\models\walk_forward_direction_bestparams_reproduction\walk_forward_oos_predictions.csv ^
  --summary artifacts\models\walk_forward_direction_bestparams_reproduction\walk_forward_summary.json ^
  --outdir artifacts\reports\lstm_walkforward_bestparams_reproduction ^
  --cost_bps 5
```

## 10. Combined LSTM + baseline comparison table

```cmd
thesis-model-comparison ^
  --baseline-metrics artifacts\reports\baseline_models_linear_svm_ablations\tables\baseline_model_metrics.csv ^
  --lstm-auc 0.550643920654932 ^
  --lstm-accuracy 0.5178571428571429 ^
  --lstm-sharpe 0.9957887190041333 ^
  --lstm-trade-rate 0.5396825396825397 ^
  --outdir artifacts\reports\model_comparison
```

## 11. Recommended reproduction sequences

From cleaned dataset:

```cmd
python -m thesis.eval.make_scientific_outputs
python -m thesis.eval.run_baseline_models_linear_svm --run-ablations --outdir artifacts\reports\baseline_models_linear_svm_ablations
thesis-model-comparison --baseline-metrics artifacts\reports\baseline_models_linear_svm_ablations\tables\baseline_model_metrics.csv --lstm-auc 0.550643920654932 --lstm-accuracy 0.5178571428571429 --lstm-sharpe 0.9957887190041333 --lstm-trade-rate 0.5396825396825397 --outdir artifacts\reports\model_comparison
```

From raw data:

```cmd
thesis-fetch-stockdata market --mode eod --symbol NVDA --start 2019-03-01 --end 2026-03-01 --csv
thesis-fetch-stockdata market --mode eod --symbol SPY --start 2019-03-01 --end 2026-03-01 --csv --outdir data\raw\macro_stock_data\SPY
thesis-fetch-stockdata market --mode eod --symbol SOXX --start 2019-03-01 --end 2026-03-01 --csv --outdir data\raw\macro_stock_data\SOXX
thesis-fetch-stockdata market --mode eod --symbol IEF --start 2019-03-01 --end 2026-03-01 --csv --outdir data\raw\macro_stock_data\IEF
thesis-fetch-stockdata news --symbols NVDA --start 2019-03-01 --end 2026-03-01 --chunk-days 30 --csv
thesis-preprocess all
python -m thesis.eval.make_scientific_outputs
python -m thesis.eval.run_baseline_models_linear_svm --run-ablations --outdir artifacts\reports\baseline_models_linear_svm_ablations
thesis-model-comparison --baseline-metrics artifacts\reports\baseline_models_linear_svm_ablations\tables\baseline_model_metrics.csv --lstm-auc 0.550643920654932 --lstm-accuracy 0.5178571428571429 --lstm-sharpe 0.9957887190041333 --lstm-trade-rate 0.5396825396825397 --outdir artifacts\reports\model_comparison
```
