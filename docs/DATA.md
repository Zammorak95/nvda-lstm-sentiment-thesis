# Data files and dataset placement

## Expected local file

The main cleaned dataset used by the modelling scripts should be placed at:

```text
data/model_feed/model_dataset_clean.csv
```

This file is the practical reproducibility anchor for the thesis workflow. It allows the reviewer to reproduce the diagnostics, classical baselines, model-comparison tables and most thesis figures without rerunning the full raw-data acquisition process.

## Repository use

This repository is intended as a transparent research companion for the thesis. It is useful for the author, supervisors, reviewers and interested readers who want to inspect the empirical workflow and reproduce the reported results.

The raw data folders and generated artifacts are ignored by Git by default to avoid accidental commits of large local files, API outputs, tokens or machine-specific files. If the cleaned dataset is shared with a reviewer, place it at the path above before running the commands.

## Data folders

```text
data/raw/          Raw downloaded inputs; ignored by Git.
data/interim/      Intermediate merged/cleaned files; ignored by Git.
data/processed/    Processed files; ignored by Git.
data/model_feed/   Final modelling dataset location.
artifacts/         Generated reports, figures, models and predictions.
```

## Minimum reproducibility dataset

For reproducing the final results without rerunning raw-data collection, the minimum required file is:

```text
data/model_feed/model_dataset_clean.csv
```

For reproducing the exact historical LSTM comparison table without retraining, the stored thesis metrics can be passed directly to `thesis-model-comparison`, as documented in `docs/EXPECTED_OUTPUTS.md`.

## Optional full raw-data workflow

When the original raw sources are available, the full workflow is:

```cmd
thesis-fetch-stockdata --help
thesis-preprocess all
```

Google Trends exports are collected manually and then placed in:

```text
data/raw/trends/
```

See `docs/DATA_ACQUISITION.md` and `docs/PREPROCESSING.md` for the full process.
