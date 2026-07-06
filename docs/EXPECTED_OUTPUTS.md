# Expected outputs

This page gives reviewers a quick checklist of what should be produced by the generic full pipeline and what the historical NVDA thesis numbers looked like.

## Main command

```bash
SYMBOL=NVDA KEYWORD="NVIDIA stock" END=2026-03-01 TRIALS=50 GPU=0 \
  bash scripts/run_stock_full_pipeline.sh main
```

## Main output folders for NVDA

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

For AMD, the same files are created with the `amd_` prefix.

## Clean thesis dataset

Expected clean dataset:

```text
data/model_feed/nvda_model_dataset_clean.csv
```

The clean dataset should contain the 16 thesis feature columns plus `date`, `target_direction` and `target_next_return`.

## LSTM outputs

Expected examples:

```text
artifacts/models/nvda_random_search_reduced_features/random_search_results.csv
artifacts/models/nvda_random_search_reduced_features/best/meta.json
artifacts/models/nvda_walk_forward_random_search_bestparams/walk_forward_oos_predictions.csv
artifacts/models/nvda_walk_forward_random_search_bestparams/walk_forward_summary.json
artifacts/models/nvda_walk_forward_random_search_bestparams/thesis_report_gross/report_summary.json
artifacts/models/nvda_walk_forward_random_search_bestparams/thesis_report_gross/figures/roc_curve.png
artifacts/models/nvda_walk_forward_random_search_bestparams/thesis_report_gross/figures/equity_drawdown.png
```

## Classical baseline outputs

Expected examples:

```text
artifacts/reports/nvda_baseline_models_linear_svm_ablations/tables/baseline_model_metrics.csv
artifacts/reports/nvda_baseline_models_linear_svm_ablations/tables/baseline_fold_metrics.csv
artifacts/reports/nvda_baseline_models_linear_svm_ablations/tables/random_forest_feature_importance_mean.csv
artifacts/reports/nvda_baseline_models_linear_svm_ablations/figures/baseline_model_auc.png
artifacts/reports/nvda_baseline_models_linear_svm_ablations/figures/baseline_cumulative_returns.png
artifacts/reports/nvda_baseline_models_linear_svm_ablations/figures/feature_set_ablation_auc.png
```

## Combined model comparison outputs

Expected examples:

```text
artifacts/reports/nvda_model_comparison/model_comparison_table.csv
artifacts/reports/nvda_model_comparison/model_comparison_table.md
artifacts/reports/nvda_model_comparison/model_comparison_table.tex
artifacts/reports/nvda_model_comparison/model_comparison_table.png
artifacts/reports/nvda_model_comparison/model_comparison_classification_table.png
artifacts/reports/nvda_model_comparison/model_comparison_trading_table.png
artifacts/reports/nvda_model_comparison/model_comparison_auc.png
```

## Historical NVDA thesis metrics

The historical final NVDA LSTM walk-forward run had approximately:

| Model | OOS AUC | Accuracy | Strategy Sharpe | Trade rate |
|---|---:|---:|---:|---:|
| Majority class | 0.5000 | 0.5575 | 1.4042 | 1.0000 |
| Logistic regression | 0.5343 | 0.4841 | 0.9590 | 0.3393 |
| Linear SVM | 0.5258 | 0.4980 | 1.2263 | 0.5278 |
| Random Forest | 0.5255 | 0.5119 | 1.1172 | 0.5813 |
| LSTM best specification | 0.5506 | 0.5179 | 0.9958 | 0.5397 |

These values are useful as a historical reference, not as hard-coded outputs. Fresh LSTM reruns can differ because neural-network training is stochastic.

## Historical 0.5506-style rerun

```bash
SYMBOL=NVDA KEYWORD="NVIDIA stock" END=2026-03-01 GPU=0 \
  bash scripts/run_stock_full_pipeline.sh legacy_05506
```

Expected output:

```text
artifacts/models/nvda_walk_forward_legacy_05506_params/
artifacts/models/nvda_walk_forward_legacy_05506_params/thesis_report_gross/
```

## Interpretation

- The LSTM historically had the highest OOS AUC among the tested models.
- The improvement over logistic regression was modest.
- Majority class can have high raw accuracy and Sharpe because it is always exposed to the dominant upward class, not because it discriminates between up and down days.
- Results should be interpreted as weak but measurable predictive signal rather than strong trading evidence.
