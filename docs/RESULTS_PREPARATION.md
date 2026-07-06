# Preparing final thesis results for GitHub

Use this after the NVDA and AMD runs are complete and after the LSTM feature-group ablation has been run.

The preparation script creates compact, GitHub-friendly evidence files:

- dataset overview and class balance;
- descriptive statistics;
- missing-value report;
- feature correlation matrix;
- correlation heatmap;
- target correlations;
- high-correlation pairs;
- LSTM feature-ablation summary rebuilt from the report JSON files;
- final results manifest.

It does not upload raw data, API credentials or trained model binaries.

## 1. Pull the latest helper script

```bash
cd ~/thesis
source .venv/bin/activate
git pull origin main_v2
```

## 2. Dry-run legacy movement first

This shows which old/fixed/debug outputs would be moved to `artifacts/legacy`:

```bash
python scripts/prepare_results_for_github.py --symbols NVDA AMD --dry-run
```

## 3. Generate final reports and move legacy outputs

```bash
python scripts/prepare_results_for_github.py --symbols NVDA AMD --move-legacy
```

The main manifest is written to:

```text
artifacts/reports/final_results_manifest.md
```

Dataset reports are written to:

```text
artifacts/reports/nvda_dataset_statistics/
artifacts/reports/amd_dataset_statistics/
```

Legacy outputs are moved to:

```text
artifacts/legacy/
```

## 4. Check what will be committed

```bash
git status --short
```

Avoid committing raw data, secrets, `.env`, `.venv`, model binaries or very large files.

## 5. Commit and push

```bash
git add artifacts/reports artifacts/models artifacts/legacy docs scripts src tests
git commit -m "Add final thesis results and dataset statistics"
git push origin main_v2
```

If Git refuses because a file is too large, inspect it with:

```bash
find artifacts -type f -size +50M -print
```

Move large non-essential files to `artifacts/legacy` or keep them local only.
