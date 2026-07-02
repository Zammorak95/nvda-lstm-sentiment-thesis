# Data policy and dataset placement

## Expected local files

The main cleaned dataset used by the modelling scripts should be placed at:

```text
data/model_feed/model_dataset_clean.csv
```

The current `.gitignore` excludes CSV and Excel files in `data/model_feed/`, as well as raw and intermediate data folders. This prevents accidental commits of proprietary or very large data.

## Can the cleaned dataset be uploaded to GitHub?

Yes, but only if all of the following are true:

1. The dataset does not contain API keys, credentials, local paths, or personal data.
2. The original data licences permit redistribution.
3. The file is small enough for normal Git use.
4. You are comfortable with others being able to download the dataset if the repository is public.

For this thesis, uploading `model_dataset_clean.csv` can improve reproducibility, but it should be treated as a deliberate publication decision rather than an accidental data commit.

## Recommended options

### Option A — Keep dataset out of Git, document expected path

This is the safest default. The repository remains code-only, and the README tells users to place the dataset at:

```text
data/model_feed/model_dataset_clean.csv
```

Use this if there is uncertainty about data licensing, news-data redistribution, or paid data sources.

### Option B — Add the clean dataset directly to Git

Use this only when the clean dataset is small and legally redistributable.

Steps:

```cmd
git add -f data\model_feed\model_dataset_clean.csv
git commit -m "Add cleaned model dataset for reproducibility"
git push
```

The `-f` is needed because CSV files in `data/model_feed/` are intentionally ignored.

### Option C — Use GitHub Releases or Git LFS for larger files

For larger datasets, avoid normal Git commits. Use GitHub Releases or Git LFS instead. The README can then point to the release asset and instruct users to download it into `data/model_feed/`.

## Data folders

```text
data/raw/          Raw downloaded inputs; ignored by Git.
data/interim/      Intermediate merged/cleaned files; ignored by Git.
data/processed/    Processed files; ignored by Git.
data/model_feed/   Final modelling dataset location; CSV/XLSX ignored by Git by default.
artifacts/         Generated reports, figures, models and predictions; ignored or treated as local output.
```

## Minimum reproducibility dataset

For reproducing the final results without rerunning the full raw-data collection process, the minimum required file is:

```text
data/model_feed/model_dataset_clean.csv
```

For reproducing the exact historical LSTM result without retraining, also preserve the historical prediction and summary files:

```text
models/walk_forward_direction_bestparams/walk_forward_oos_predictions_bestparams.csv
models/walk_forward_direction_bestparams/walk_forward_summary_best_features.json
```

These model artifacts are currently ignored by Git by default. If you want them public, place them in a release or add a small curated reproduction package.
