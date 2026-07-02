@echo off
setlocal

REM Reproduce the main thesis outputs from a clean Windows CMD environment.
REM Run from the repository root.

cd /d "%~dp0\.."

echo === Creating virtual environment if needed ===
if not exist .venv\Scripts\python.exe (
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo === Installing project dependencies ===
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"

echo === Checking dataset ===
if not exist data\model_feed\model_dataset_clean.csv (
    echo Missing data\model_feed\model_dataset_clean.csv
    echo Place the cleaned dataset there and rerun this script.
    exit /b 1
)

echo === Scientific dataset outputs ===
python -m thesis.eval.make_scientific_outputs

echo === Classical baselines with linear SVM and ablations ===
python -m thesis.eval.run_baseline_models_linear_svm --run-ablations --outdir artifacts\reports\baseline_models_linear_svm_ablations

echo === Combined model comparison table using historical final LSTM metrics ===
thesis-model-comparison ^
  --baseline-metrics artifacts\reports\baseline_models_linear_svm_ablations\tables\baseline_model_metrics.csv ^
  --lstm-auc 0.550643920654932 ^
  --lstm-accuracy 0.5178571428571429 ^
  --lstm-sharpe 0.9957887190041333 ^
  --lstm-trade-rate 0.5396825396825397 ^
  --outdir artifacts\reports\model_comparison

echo === Done ===
echo Main outputs:
echo   artifacts\reports\scientific_outputs
echo   artifacts\reports\baseline_models_linear_svm_ablations
echo   artifacts\reports\model_comparison

endlocal
