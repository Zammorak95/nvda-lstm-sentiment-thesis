# Command reference

This document lists the canonical commands for reproducing the thesis workflow. Run commands from the repository root.

## 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python -m pip install tensorflow
```

For a pinned non-TensorFlow environment:

```bash
python -m pip install -r requirements-reproducibility.txt
python -m pip install -e ".[dev]"
```

TensorFlow is kept separate because installation differs by OS/GPU.

## 2. Main generic end-to-end pipeline

Check configuration:

```bash
ROOT=$PWD PYTHON=python bash scripts/run_stock_full_pipeline.sh env
```

Run the main NVIDIA thesis pipeline:

```bash
SYMBOL=NVDA KEYWORD="NVIDIA stock" END=2026-03-01 TRIALS=50 GPU=0 \
  bash scripts/run_stock_full_pipeline.sh main
```

Run the full NVIDIA pipeline including fixed comparison runs:

```bash
SYMBOL=NVDA KEYWORD="NVIDIA stock" END=2026-03-01 TRIALS=50 GPU=0 \
  bash scripts/run_stock_full_pipeline.sh all
```

Run the AMD robustness pipeline:

```bash
SYMBOL=AMD KEYWORD="AMD stock" END=2026-02-26 TRIALS=50 GPU=0 \
  bash scripts/run_stock_full_pipeline.sh all
```

Main route:

```text
data -> reduced -> random_search -> walk_bestparams -> report_bestparams -> baselines -> model_comparison -> summary
```

## 3. Pipeline phases

```bash
SYMBOL=NVDA KEYWORD="NVIDIA stock" END=2026-03-01 bash scripts/run_stock_full_pipeline.sh data
SYMBOL=NVDA KEYWORD="NVIDIA stock" bash scripts/run_stock_full_pipeline.sh reduced
SYMBOL=NVDA KEYWORD="NVIDIA stock" TRIALS=50 GPU=0 bash scripts/run_stock_full_pipeline.sh random_search
SYMBOL=NVDA KEYWORD="NVIDIA stock" GPU=0 bash scripts/run_stock_full_pipeline.sh walk_bestparams
SYMBOL=NVDA KEYWORD="NVIDIA stock" bash scripts/run_stock_full_pipeline.sh report_bestparams
SYMBOL=NVDA KEYWORD="NVIDIA stock" bash scripts/run_stock_full_pipeline.sh baselines
SYMBOL=NVDA KEYWORD="NVIDIA stock" bash scripts/run_stock_full_pipeline.sh model_comparison
SYMBOL=NVDA KEYWORD="NVIDIA stock" bash scripts/run_stock_full_pipeline.sh summary
```

## 4. Historical NVDA 0.5506-style check

The old thesis result around OOS AUC `0.5506` came from this specification:

```text
lookback=90, lstm_units=96, dense_units=64, lr=0.0003,
batch=64, dropout=0.10, recurrent_dropout=0.20, auto_threshold=True
```

Run it on the current clean dataset with:

```bash
SYMBOL=NVDA KEYWORD="NVIDIA stock" END=2026-03-01 GPU=0 \
  bash scripts/run_stock_full_pipeline.sh legacy_05506
```

This reruns the old parameter setting. It does not hard-code the old metric, so small differences are expected.

## 5. Raw-data acquisition details

The `data` phase performs raw checks/fetching automatically. It uses:

- StockData.org EOD data for the target ticker and SPY/SOXX/IEF;
- StockData.org symbol-filtered news headlines;
- PyTrends Google Trends collection using the configured `KEYWORD`.

Default news limit:

```bash
NEWS_LIMIT_PER_DAY=25
```

If local raw/processed files already exist, the pipeline reuses them. Set `FORCE=1` to rebuild/refetch.

## 6. Classical baselines only

```bash
SYMBOL=NVDA KEYWORD="NVIDIA stock" bash scripts/run_stock_full_pipeline.sh baselines
```

Direct command equivalent:

```bash
python -m thesis.eval.run_baseline_models_linear_svm \
  --dataset data/model_feed/nvda_model_dataset_clean.csv \
  --run-ablations \
  --outdir artifacts/reports/nvda_baseline_models_linear_svm_ablations
```

## 7. Combined model comparison only

```bash
SYMBOL=NVDA KEYWORD="NVIDIA stock" bash scripts/run_stock_full_pipeline.sh model_comparison
```

Direct command equivalent:

```bash
python -m thesis.eval.make_model_comparison_table \
  --baseline-metrics artifacts/reports/nvda_baseline_models_linear_svm_ablations/tables/baseline_model_metrics.csv \
  --lstm-summary artifacts/models/nvda_walk_forward_random_search_bestparams/thesis_report_gross/report_summary.json \
  --outdir artifacts/reports/nvda_model_comparison
```

## 8. Main outputs

For NVDA, the main generated files/folders are:

```text
data/model_feed/nvda_model_dataset.csv
data/model_feed/nvda_model_dataset_clean.csv
data/model_feed/nvda_model_dataset_audit.xlsx
artifacts/models/nvda_random_search_reduced_features/
artifacts/models/nvda_walk_forward_random_search_bestparams/
artifacts/models/nvda_walk_forward_random_search_bestparams/thesis_report_gross/
artifacts/reports/nvda_baseline_models_linear_svm_ablations/
artifacts/reports/nvda_model_comparison/
artifacts/reports/nvda_full_pipeline_summary.csv
```

## 9. Lightweight checks

```bash
pytest tests/test_imports.py
ROOT=$PWD PYTHON=python bash scripts/run_stock_full_pipeline.sh env
python src/thesis/pipelines/run_stock_pipeline.py --help
python src/thesis/pipelines/run_stock_pipeline_auto_window.py --help
```
