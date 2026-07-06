# Final thesis results manifest

Generated: 2026-07-05T22:06:37

This file lists the compact results and statistical reports that should be kept in GitHub. Raw data, API credentials and trained model binaries should remain local.

## NVDA

### Core model results
- `artifacts/reports/nvda_full_pipeline_summary.csv`
- `artifacts/reports/nvda_model_comparison/model_comparison_table.csv`
- `artifacts/reports/nvda_model_comparison/model_comparison_auc.png`
- `artifacts/reports/nvda_model_comparison/model_comparison_classification_table.png`
- `artifacts/reports/nvda_model_comparison/model_comparison_trading_table.png`
- `artifacts/reports/nvda_baseline_models_linear_svm_ablations/tables/baseline_model_metrics.csv`
- `artifacts/reports/nvda_baseline_models_linear_svm_ablations/tables/baseline_fold_metrics.csv`
- `artifacts/reports/nvda_baseline_models_linear_svm_ablations/figures/feature_set_ablation_auc.png`
- `artifacts/models/nvda_walk_forward_random_search_bestparams/walk_forward_summary.json`
- `artifacts/models/nvda_walk_forward_random_search_bestparams/thesis_report_gross/report_summary.json`
- `artifacts/models/nvda_walk_forward_random_search_bestparams/thesis_report_gross/figures/roc_curve.png`
- `artifacts/models/nvda_walk_forward_random_search_bestparams/thesis_report_gross/figures/equity_drawdown.png`
- `artifacts/models/nvda_lstm_feature_ablation/lstm_feature_ablation_summary.csv`
- `artifacts/models/nvda_lstm_feature_ablation/lstm_feature_ablation_summary_fixed.csv`

### Dataset statistics
- `artifacts/reports/nvda_dataset_statistics/dataset_overview.csv`
- `artifacts/reports/nvda_dataset_statistics/descriptive_statistics.csv`
- `artifacts/reports/nvda_dataset_statistics/correlation_matrix.csv`
- `artifacts/reports/nvda_dataset_statistics/correlation_heatmap.png`
- `artifacts/reports/nvda_dataset_statistics/target_correlations.csv`
- `artifacts/reports/nvda_dataset_statistics/high_correlations.csv`
- `artifacts/reports/nvda_dataset_statistics/feature_group_summary.csv`

## AMD

### Core model results
- `artifacts/reports/amd_full_pipeline_summary.csv`
- `artifacts/reports/amd_model_comparison/model_comparison_table.csv`
- `artifacts/reports/amd_model_comparison/model_comparison_auc.png`
- `artifacts/reports/amd_model_comparison/model_comparison_classification_table.png`
- `artifacts/reports/amd_model_comparison/model_comparison_trading_table.png`
- `artifacts/reports/amd_baseline_models_linear_svm_ablations/tables/baseline_model_metrics.csv`
- `artifacts/reports/amd_baseline_models_linear_svm_ablations/tables/baseline_fold_metrics.csv`
- `artifacts/reports/amd_baseline_models_linear_svm_ablations/figures/feature_set_ablation_auc.png`
- `artifacts/models/amd_walk_forward_random_search_bestparams/walk_forward_summary.json`
- `artifacts/models/amd_walk_forward_random_search_bestparams/thesis_report_gross/report_summary.json`
- `artifacts/models/amd_walk_forward_random_search_bestparams/thesis_report_gross/figures/roc_curve.png`
- `artifacts/models/amd_walk_forward_random_search_bestparams/thesis_report_gross/figures/equity_drawdown.png`

### Dataset statistics
- `artifacts/reports/amd_dataset_statistics/dataset_overview.csv`
- `artifacts/reports/amd_dataset_statistics/descriptive_statistics.csv`
- `artifacts/reports/amd_dataset_statistics/correlation_matrix.csv`
- `artifacts/reports/amd_dataset_statistics/correlation_heatmap.png`
- `artifacts/reports/amd_dataset_statistics/target_correlations.csv`
- `artifacts/reports/amd_dataset_statistics/high_correlations.csv`
- `artifacts/reports/amd_dataset_statistics/feature_group_summary.csv`

## Preparation status

### Dataset reports
- NVDA: ok — `/home/zammorak/thesis/artifacts/reports/nvda_dataset_statistics`
- AMD: ok — `/home/zammorak/thesis/artifacts/reports/amd_dataset_statistics`

### LSTM ablation summaries
- NVDA: ok — `/home/zammorak/thesis/artifacts/models/nvda_lstm_feature_ablation/lstm_feature_ablation_summary.csv`
- AMD: missing — `/home/zammorak/thesis/artifacts/models/amd_lstm_feature_ablation`

## Suggested Git commands

```bash
git status --short
git add artifacts/reports artifacts/models artifacts/legacy docs scripts src tests
git commit -m "Add final thesis results and dataset statistics"
git push origin main_v2
```
