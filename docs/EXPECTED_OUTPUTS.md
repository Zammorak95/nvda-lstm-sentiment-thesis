# Expected outputs

This page gives reviewers a quick checklist of what should be produced by the main workflow and what the key thesis numbers are expected to look like.

## Main output folders

After running the recommended workflow, inspect:

```text
artifacts/reports/scientific_outputs/
artifacts/reports/baseline_models_linear_svm_ablations/
artifacts/models/walk_forward_direction_bestparams_reproduction/
artifacts/reports/model_comparison/
```

## Dataset diagnostics

Command:

```cmd
python -m thesis.eval.make_scientific_outputs
```

Expected examples:

```text
artifacts/reports/scientific_outputs/tables/dataset_overview.csv
artifacts/reports/scientific_outputs/tables/target_class_balance.csv
artifacts/reports/scientific_outputs/tables/feature_descriptives.csv
artifacts/reports/scientific_outputs/figures/target_class_distribution.png
artifacts/reports/scientific_outputs/figures/feature_correlation_heatmap.png
```

## Classical baseline outputs

Command:

```cmd
python -m thesis.eval.run_baseline_models_linear_svm --run-ablations --outdir artifacts\reports\baseline_models_linear_svm_ablations
```

Expected examples:

```text
artifacts/reports/baseline_models_linear_svm_ablations/tables/baseline_model_metrics.csv
artifacts/reports/baseline_models_linear_svm_ablations/tables/baseline_fold_metrics.csv
artifacts/reports/baseline_models_linear_svm_ablations/tables/random_forest_feature_importance_mean.csv
artifacts/reports/baseline_models_linear_svm_ablations/figures/baseline_model_auc.png
artifacts/reports/baseline_models_linear_svm_ablations/figures/baseline_cumulative_returns.png
```

## Combined model comparison outputs

Command:

```cmd
thesis-model-comparison ^
  --baseline-metrics artifacts\reports\baseline_models_linear_svm_ablations\tables\baseline_model_metrics.csv ^
  --lstm-auc 0.550643920654932 ^
  --lstm-accuracy 0.5178571428571429 ^
  --lstm-sharpe 0.9957887190041333 ^
  --lstm-trade-rate 0.5396825396825397 ^
  --outdir artifacts\reports\model_comparison
```

Expected files:

```text
artifacts/reports/model_comparison/model_comparison_table.csv
artifacts/reports/model_comparison/model_comparison_table.md
artifacts/reports/model_comparison/model_comparison_table.tex
artifacts/reports/model_comparison/model_comparison_table.png
artifacts/reports/model_comparison/model_comparison_classification_table.png
artifacts/reports/model_comparison/model_comparison_trading_table.png
artifacts/reports/model_comparison/model_comparison_auc.png
```

## Key expected thesis metrics

The historical final LSTM walk-forward run used in the thesis comparison has approximately:

| Model | OOS AUC | Accuracy | Strategy Sharpe | Trade rate |
|---|---:|---:|---:|---:|
| Majority class | 0.5000 | 0.5575 | 1.4042 | 1.0000 |
| Logistic regression | 0.5343 | 0.4841 | 0.9590 | 0.3393 |
| Linear SVM | 0.5258 | 0.4980 | 1.2263 | 0.5278 |
| Random Forest | 0.5255 | 0.5119 | 1.1172 | 0.5813 |
| LSTM best specification | 0.5506 | 0.5179 | 0.9958 | 0.5397 |

Interpretation:

- The LSTM has the highest OOS AUC among the tested models.
- The improvement over logistic regression is modest.
- Majority class has high raw accuracy and Sharpe because it is always exposed to the dominant upward class, not because it discriminates between up and down days.
- Results should be interpreted as weak but measurable predictive signal rather than strong trading evidence.

## Reproduction tolerance

Classical baseline outputs should reproduce closely when the same dataset and package versions are used. LSTM retraining may differ slightly across machines because TensorFlow training can vary across CPU/GPU, operating system and random initialization.
