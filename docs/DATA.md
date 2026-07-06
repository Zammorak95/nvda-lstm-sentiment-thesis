# Data files and dataset placement

## Expected clean datasets

The generic pipeline writes ticker-specific cleaned modelling datasets, for example:

```text
data/model_feed/nvda_model_dataset_clean.csv
data/model_feed/amd_model_dataset_clean.csv
```

These files are the practical reproducibility anchors for thesis review. They allow reviewers to reproduce baselines, model-comparison tables and most thesis figures without rerunning raw-data collection.

## Repository use

This repository is intended as a transparent research companion for the thesis. It is useful for the author, supervisors, reviewers and interested readers who want to inspect the empirical workflow and reproduce the reported results.

The raw data folders and generated artifacts are ignored by Git by default to avoid accidental commits of large local files, API outputs, local environment files or machine-specific files.

## Data folders

```text
data/raw/          Raw downloaded inputs and PyTrends checkpoints.
data/interim/      Intermediate merged/cleaned files.
data/processed/    Processed EOD, sentiment and Trends files.
data/model_feed/   Final ticker-specific model datasets and audit workbooks.
artifacts/         Generated reports, figures, models and predictions.
```

## Minimum review dataset

For reproducing the final outputs without rerunning raw-data collection, the minimum required file is the relevant clean dataset, for example:

```text
data/model_feed/nvda_model_dataset_clean.csv
```

## Full raw-to-results workflow

```bash
SYMBOL=NVDA KEYWORD="NVIDIA stock" END=2026-03-01 TRIALS=50 GPU=0 \
  bash scripts/run_stock_full_pipeline.sh main
```

This performs raw checks/fetching, preprocessing, random search, LSTM walk-forward evaluation, baseline ablations, model comparison and summary generation.

See `docs/DATA_ACQUISITION.md`, `docs/RAW_DATA_PIPELINE.md`, `docs/EXPECTED_OUTPUTS.md` and `docs/SCRIPT_INDEX.md` for details.
