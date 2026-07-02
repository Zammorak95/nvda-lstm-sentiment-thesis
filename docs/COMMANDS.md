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

## 2. Data location

The expected cleaned model dataset is:

```text
data/model_feed/model_dataset_clean.csv
```

The dataset is intentionally ignored by Git by default. See `docs/DATA.md` for guidance on whether to publish it.

## 3. Dataset diagnostics and thesis figures

```cmd
python -m thesis.eval.make_scientific_outputs
```

Output:

```text
artifacts/reports/scientific_outputs/
```

## 4. Classical baselines with linear SVM

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

## 5. LSTM hyperparameter tuning

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

## 6. Final LSTM walk-forward reproduction

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

## 7. LSTM statistical report

```cmd
python src\thesis\eval\thesis_walkforward_report.py ^
  --oos artifacts\models\walk_forward_direction_bestparams_reproduction\walk_forward_oos_predictions.csv ^
  --summary artifacts\models\walk_forward_direction_bestparams_reproduction\walk_forward_summary.json ^
  --outdir artifacts\reports\lstm_walkforward_bestparams_reproduction ^
  --cost_bps 5
```

## 8. Combined LSTM + baseline comparison table

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

## 9. Recommended reproduction sequence

```cmd
python -m thesis.eval.make_scientific_outputs
python -m thesis.eval.run_baseline_models_linear_svm --run-ablations --outdir artifacts\reports\baseline_models_linear_svm_ablations
thesis-walkforward-lstm --data data\model_feed\model_dataset_clean.csv --outdir artifacts\models\walk_forward_direction_bestparams_reproduction --lookback 90 --initial_train 700 --val_size 126 --test_horizon 63 --step 63 --epochs 30 --batch 64 --lr 0.0003 --lstm_units 96 --dense_units 64 --dropout 0.10 --recurrent_dropout 0.20 --auto_threshold
thesis-model-comparison --baseline-metrics artifacts\reports\baseline_models_linear_svm_ablations\tables\baseline_model_metrics.csv --lstm-auc 0.550643920654932 --lstm-accuracy 0.5178571428571429 --lstm-sharpe 0.9957887190041333 --lstm-trade-rate 0.5396825396825397 --outdir artifacts\reports\model_comparison
```
