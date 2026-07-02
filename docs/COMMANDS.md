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

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

Install TensorFlow only when reproducing the LSTM runs:

```bash
python -m pip install tensorflow statsmodels tabulate
```

For AMD ROCm/Linux, install the TensorFlow build appropriate for the local ROCm setup instead of the generic CPU package.

## 2. Raw/intermediate data location

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

## 3. Preprocessing pipeline

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

Output:

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

## 4. Dataset diagnostics and thesis figures

```cmd
python -m thesis.eval.make_scientific_outputs
```

Output:

```text
artifacts/reports/scientific_outputs/
```

## 5. Classical baselines with linear SVM

Main benchmark run:

```cmd
python -m thesis.eval.run_baseline_models_linear_svm ^
  --outdir artifacts\reports\baseline_models_linear_svm
```

Feature-set ablation run:

```cmd
python -m thesis.eval.run_baseline_models_linear_svm ^
  --run-ablations ^
  --outdir artifacts\reports\baseline_models_linear_svm_ablations
```

Output:

```text
artifacts/reports/baseline_models_linear_svm/
artifacts/reports/baseline_models_linear_svm_ablations/
```

## 6. LSTM hyperparameter tuning

Short technical smoke test:

```cmd
thesis-tune-lstm --trials 2 --max_epochs 2 --auto_threshold
```

Full random-search run:

```cmd
thesis-tune-lstm --trials 50 --auto_threshold
```

Equivalent legacy script:

```cmd
python src\thesis\model_training\optimalisation\random_search_lstm_direction_v2.py --trials 50 --auto_threshold
```

## 7. Final LSTM walk-forward reproduction

The thesis best-parameter reproduction command is:

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

On Linux/ROCm, add `--gpu 0` when the ROCm GPU environment is configured.

## 8. LSTM statistical report

```cmd
python src\thesis\eval\thesis_walkforward_report.py ^
  --oos artifacts\models\walk_forward_direction_bestparams_reproduction\walk_forward_oos_predictions.csv ^
  --summary artifacts\models\walk_forward_direction_bestparams_reproduction\walk_forward_summary.json ^
  --outdir artifacts\reports\lstm_walkforward_bestparams_reproduction ^
  --cost_bps 5
```

## 9. Combined LSTM + baseline comparison table

Use the stored final LSTM values from the thesis run:

```cmd
thesis-model-comparison ^
  --baseline-metrics artifacts\reports\baseline_models_linear_svm_ablations\tables\baseline_model_metrics.csv ^
  --lstm-auc 0.550643920654932 ^
  --lstm-accuracy 0.5178571428571429 ^
  --lstm-sharpe 0.9957887190041333 ^
  --lstm-trade-rate 0.5396825396825397 ^
  --outdir artifacts\reports\model_comparison
```

Output:

```text
artifacts/reports/model_comparison/model_comparison_table.csv
artifacts/reports/model_comparison/model_comparison_classification_table.png
artifacts/reports/model_comparison/model_comparison_trading_table.png
artifacts/reports/model_comparison/model_comparison_auc.png
```

## 10. Recommended reproduction sequence

From cleaned dataset:

```cmd
python -m thesis.eval.make_scientific_outputs
python -m thesis.eval.run_baseline_models_linear_svm --run-ablations --outdir artifacts\reports\baseline_models_linear_svm_ablations
thesis-walkforward-lstm --data data\model_feed\model_dataset_clean.csv --outdir artifacts\models\walk_forward_direction_bestparams_reproduction --lookback 90 --initial_train 700 --val_size 126 --test_horizon 63 --step 63 --epochs 30 --batch 64 --lr 0.0003 --lstm_units 96 --dense_units 64 --dropout 0.10 --recurrent_dropout 0.20 --auto_threshold
thesis-model-comparison --baseline-metrics artifacts\reports\baseline_models_linear_svm_ablations\tables\baseline_model_metrics.csv --lstm-auc 0.550643920654932 --lstm-accuracy 0.5178571428571429 --lstm-sharpe 0.9957887190041333 --lstm-trade-rate 0.5396825396825397 --outdir artifacts\reports\model_comparison
```

From raw files:

```cmd
thesis-preprocess all
python -m thesis.eval.make_scientific_outputs
python -m thesis.eval.run_baseline_models_linear_svm --run-ablations --outdir artifacts\reports\baseline_models_linear_svm_ablations
thesis-model-comparison --baseline-metrics artifacts\reports\baseline_models_linear_svm_ablations\tables\baseline_model_metrics.csv --lstm-auc 0.550643920654932 --lstm-accuracy 0.5178571428571429 --lstm-sharpe 0.9957887190041333 --lstm-trade-rate 0.5396825396825397 --outdir artifacts\reports\model_comparison
```
