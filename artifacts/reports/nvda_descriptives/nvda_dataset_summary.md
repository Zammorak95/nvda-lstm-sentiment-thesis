# NVDA dataset descriptive summary

- Dataset: `data/model_feed/nvda_model_dataset_clean.csv`
- Observations: 1,284
- Input features: 16
- Missing values in input features: 0
- Date range: 2021-01-04 to 2026-02-25
- Class balance: 689 up days (53.66%); 595 down days (46.34%)

## Highest Pearson feature correlations
- spy_return and soxx_return: 0.8179
- log_return and soxx_return: 0.8103
- trends_zscore_30d and trends_spike: 0.7370

## Largest target correlations
- trends_zscore_30d: corr_target_direction=0.0183, corr_target_next_return=0.0755
- trends_spike: corr_target_direction=0.0288, corr_target_next_return=0.0661
- soxx_return: corr_target_direction=-0.0014, corr_target_next_return=-0.0482
- log_return: corr_target_direction=-0.0005, corr_target_next_return=-0.0432
- volume_change: corr_target_direction=0.0008, corr_target_next_return=0.0430
- ief_return: corr_target_direction=0.0418, corr_target_next_return=0.0012
- volatility_20d: corr_target_direction=-0.0404, corr_target_next_return=-0.0337
- spy_return: corr_target_direction=-0.0140, corr_target_next_return=-0.0381
- momentum_5d: corr_target_direction=-0.0377, corr_target_next_return=-0.0353
- trends_momentum_7d: corr_target_direction=0.0008, corr_target_next_return=0.0373