# NVDA LSTM Sentiment Thesis

This repository contains the code used for a Master's thesis on forecasting NVIDIA (NVDA) next-day stock-price direction using LSTM models, market variables, news sentiment, Google Trends attention variables and classical machine-learning benchmarks.

The repository is structured to make the empirical workflow reproducible: from the cleaned modelling dataset to dataset diagnostics, benchmark models, LSTM walk-forward evaluation and final thesis-ready tables/figures.

## Research workflow

The empirical workflow is:

```text
Raw sources
  -> preprocessing and feature engineering
  -> data/model_feed/model_dataset_clean.csv
  -> dataset diagnostics and feature assessment
  -> classical benchmark models
  -> LSTM walk-forward validation
  -> combined model-comparison tables and figures
```

The most important reproducibility input is:

```text
data/model_feed/model_dataset_clean.csv
```

## Quickstart: reproduce the main tables and figures

### 1. Clone and select the branch

```cmd
git clone https://github.com/Zammorak95/nvda-lstm-sentiment-thesis.git
cd nvda-lstm-sentiment-thesis
git switch main_v2
```

### 2. Create a virtual environment

Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

TensorFlow is only needed for rerunning LSTM training:

```cmd
python -m pip install tensorflow statsmodels tabulate
```

### 3. Add the cleaned dataset

Place the cleaned dataset at:

```text
data/model_feed/model_dataset_clean.csv
```

CSV files in `data/model_feed/` are ignored by Git by default. See [Data policy](docs/DATA.md) for guidance on publishing the clean dataset.

### 4. Generate dataset diagnostics

```cmd
python -m thesis.eval.make_scientific_outputs
```

Output:

```text
artifacts/reports/scientific_outputs/
```

### 5. Run classical baselines with linear SVM

```cmd
python -m thesis.eval.run_baseline_models_linear_svm ^
  --run-ablations ^
  --outdir artifacts\reports\baseline_models_linear_svm_ablations
```

This evaluates:

- Majority-class benchmark
- Logistic regression
- Linear SVM
- Random Forest

Output:

```text
artifacts/reports/baseline_models_linear_svm_ablations/
```

### 6. Reproduce the final LSTM walk-forward specification

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

On Linux/ROCm, add `--gpu 0` if the ROCm GPU environment is configured.

### 7. Generate combined LSTM + benchmark comparison tables

To reproduce the thesis table using the historical final LSTM metrics:

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
artifacts/reports/model_comparison/
```

The most useful thesis files are:

```text
model_comparison_classification_table.png
model_comparison_trading_table.png
model_comparison_auc.png
model_comparison_table.csv
```

## One-command Windows reproduction

After placing `data/model_feed/model_dataset_clean.csv`, run:

```cmd
scripts\reproduce_windows.cmd
```

This creates the virtual environment if needed, installs the project, runs dataset diagnostics, runs the linear-SVM baseline ablation, and generates the combined model-comparison table using the historical final LSTM metrics.

## Canonical command list

See [Command reference](docs/COMMANDS.md).

Common commands:

```cmd
python -m thesis.eval.make_scientific_outputs
python -m thesis.eval.run_baseline_models_linear_svm --run-ablations
thesis-tune-lstm --trials 50 --auto_threshold
thesis-walkforward-lstm --help
thesis-model-comparison --help
```

## Project layout

```text
.
├── data/
│   └── model_feed/                 # Expected location for model_dataset_clean.csv
├── docs/
│   ├── COMMANDS.md                  # Canonical command reference
│   ├── DATA.md                      # Dataset and publication policy
│   ├── REPRODUCIBILITY.md           # End-to-end reproduction guide
│   └── SCRIPT_INDEX.md              # Active scripts and naming conventions
├── scripts/
│   └── reproduce_windows.cmd        # Windows reproduction helper
├── src/thesis/
│   ├── eval/                        # Reporting, baselines, tables and figures
│   └── model_training/              # LSTM training and walk-forward evaluation
├── artifacts/                       # Generated outputs, ignored/local
├── Makefile
├── pyproject.toml
└── README.md
```

## Active scripts

The preferred entry points are documented in [Script index](docs/SCRIPT_INDEX.md).

Main active commands:

| Purpose | Command |
|---|---|
| Dataset diagnostics | `python -m thesis.eval.make_scientific_outputs` |
| Classical baselines | `python -m thesis.eval.run_baseline_models_linear_svm` |
| Feature ablation | `python -m thesis.eval.run_baseline_models_linear_svm --run-ablations` |
| LSTM tuning | `thesis-tune-lstm` |
| LSTM walk-forward evaluation | `thesis-walkforward-lstm` |
| Combined comparison table | `thesis-model-comparison` |

Legacy scripts are kept for compatibility, but the commands above should be used in new reproduction runs.

## Reproducibility notes

The classical benchmark models are deterministic or seeded and should reproduce closely given the same dataset and package versions. The LSTM training can vary slightly across runs because TensorFlow training is stochastic and can differ between CPU/GPU, operating systems and hardware.

The final reported LSTM result in the thesis should be based on walk-forward out-of-sample predictions, not on random-search validation performance.

## Data publication

The clean modelling dataset can be uploaded to GitHub only if the underlying data licences allow redistribution and the file is small enough for normal Git use. Otherwise, use GitHub Releases, Git LFS, or keep the dataset private and document the required path. See [Data policy](docs/DATA.md).
