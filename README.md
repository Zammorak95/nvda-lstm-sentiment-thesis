# NVDA LSTM Thesis

This repository contains the code used for a Master's thesis on next-day stock-direction prediction with LSTM models, market variables, news sentiment and Google Trends attention variables. The original case study is NVIDIA (NVDA), with AMD as an additional robustness check.

The current workflow is designed to be reproducible from raw data to final model summaries:

```text
StockData.org EOD prices + StockData.org news + Google Trends via PyTrends
  -> raw/intermediate checkpoint files
  -> preprocessing and feature engineering
  -> model dataset and audit workbook
  -> thesis 16-feature clean dataset
  -> random-search hyperparameter optimization
  -> walk-forward LSTM using the best random-search hyperparameters
  -> gross LSTM report
  -> classical baselines and feature ablations
  -> combined LSTM + benchmark comparison tables
  -> summary CSV
```

## Main end-to-end pipeline

Use the generic pipeline for both NVDA and AMD. The only stock-specific inputs are the ticker, Google Trends search term, optional scan start and end date.

```bash
cd /home/zammorak/thesis
source .venv/bin/activate
set -a
source .env
set +a
```

Run the full NVIDIA pipeline:

```bash
SYMBOL=NVDA KEYWORD="NVIDIA stock" END=2026-03-01 TRIALS=50 GPU=0 \
  bash scripts/run_stock_full_pipeline.sh all
```

Run the full AMD robustness pipeline:

```bash
SYMBOL=AMD KEYWORD="AMD stock" END=2026-02-26 TRIALS=50 GPU=0 \
  bash scripts/run_stock_full_pipeline.sh all
```

The main thesis route is:

```text
data -> reduced -> random_search -> walk_bestparams -> report_bestparams -> baselines -> model_comparison -> summary
```

The same script can be run in phases:

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

Optional fixed-parameter comparison runs can also be executed:

```bash
SYMBOL=NVDA KEYWORD="NVIDIA stock" GPU=0 bash scripts/run_stock_full_pipeline.sh fixed_compare
```

## Historical NVDA result around AUC 0.5506

The historical thesis run that produced an OOS AUC of approximately `0.5506` used the following LSTM specification:

```text
lookback=90
lstm_units=96
dense_units=64
learning_rate=0.0003
batch=64
dropout=0.10
recurrent_dropout=0.20
auto_threshold=True
```

The generic random-search space now includes `lstm_units=96`, so the current full pipeline can rediscover or approximate that specification. For a direct fixed-parameter check, run:

```bash
SYMBOL=NVDA KEYWORD="NVIDIA stock" END=2026-03-01 GPU=0 \
  bash scripts/run_stock_full_pipeline.sh legacy_05506
```

This does not hard-code the old result; it reruns the old parameter setting on the current clean dataset. Small differences are expected because TensorFlow training is stochastic and because the dataset/window may differ from the historical run.

## API-conscious raw-data handling

The pipeline avoids unnecessary API calls:

- if processed EOD files already exist, it skips EOD API fetching;
- if raw EOD CSVs already exist, it reuses and processes them;
- if processed Google Trends exists, it skips PyTrends;
- if intermediate Google Trends chunks exist, it resumes from them;
- if raw news CSVs or processed daily sentiment already exist, it does not call the StockData news API;
- if `NEWS_START=auto`, it first tries to infer the start date from local news/sentiment files and only scans StockData when no local start date is available.

The default news start is based on the earliest locally available or scanned StockData news date for the ticker. The default end date is yesterday, but thesis reproduction should usually pass an explicit `END`.

## Important outputs

For a ticker such as `NVDA`, the main outputs are:

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

For AMD, the same structure is created with the `amd_` prefix.

## Configuration

Common environment variables:

```text
SYMBOL                  Stock ticker, e.g. NVDA or AMD
KEYWORD                 Google Trends term, e.g. "NVIDIA stock" or "AMD stock"
SCAN_START              Earliest date to test for news availability, default 2018-01-01
END                     Final date for the data window; default is yesterday
NEWS_START              auto or explicit YYYY-MM-DD
NEWS_LIMIT_PER_DAY      StockData news articles requested per day; default 25
MARKET_BUFFER_DAYS      Extra market/Trends history before news_start, default 45
TRIALS                  Random-search trials, default 50
RANDOM_EPOCHS           Max epochs per random-search trial, default 50
WALK_EPOCHS             Epochs for final walk-forward training, default 30
GPU                     ROCm GPU index, usually 0
FORCE                   Set to 1 to rebuild/refetch even if files exist
```

## Repository setup

Clone and select the working branch:

```bash
git clone https://github.com/Zammorak95/nvda-lstm-sentiment-thesis.git
cd nvda-lstm-sentiment-thesis
git switch main_v2
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python -m pip install tensorflow
```

Set the StockData.org token locally as `STOCKDATA_API_TOKEN` or `STOCKDATA_API_KEY`. Use `.env.example` as a template.

Recommended local `.env` content:

```text
STOCKDATA_API_TOKEN=your_stockdata_token_here
NEWS_LIMIT_PER_DAY=25
```

## Project layout

```text
.
├── data/                         # Local raw/intermediate/processed/model-feed data
├── docs/                         # Thesis workflow notes and command references
├── scripts/
│   └── run_stock_full_pipeline.sh # Active generic end-to-end stock pipeline
├── src/thesis/
│   ├── pipelines/                # Generic stock data pipeline wrappers
│   ├── eval/                     # Reports, tables, figures and statistical evaluation
│   └── model_training/           # Random search and walk-forward LSTM scripts
├── artifacts/                    # Generated reports/models/results
├── requirements-reproducibility.txt
├── .env.example
├── Makefile
├── pyproject.toml
└── README.md
```
