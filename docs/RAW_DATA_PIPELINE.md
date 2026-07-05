# Raw-data pipeline notes

The practical review path starts from the ticker-specific cleaned modelling dataset:

```text
data/model_feed/nvda_model_dataset_clean.csv
data/model_feed/amd_model_dataset_clean.csv
```

A full rebuild is available through the generic pipeline.

## Conceptual raw-data flow

```text
1. StockData.org market data
   -> raw target/SPY/SOXX/IEF OHLCV files
   -> processed daily return, momentum, volatility and volume features

2. StockData.org news headlines
   -> raw financial-news headline files
   -> cleaned and trading-day aligned headline data
   -> VADER sentiment features

3. Google Trends via PyTrends
   -> full-period reference series
   -> shorter daily chunks
   -> scaled and chain-linked daily attention series
   -> z-score, momentum and spike features

4. Merge step
   stock features + sentiment features + trends features + macro/ETF features
   -> data/model_feed/<symbol>_model_dataset.csv
   -> data/model_feed/<symbol>_model_dataset_clean.csv
```

## Main command

```bash
SYMBOL=NVDA KEYWORD="NVIDIA stock" END=2026-03-01 TRIALS=50 GPU=0 \
  bash scripts/run_stock_full_pipeline.sh main
```

The `data` phase performs raw checks/fetching and preprocessing:

```bash
SYMBOL=NVDA KEYWORD="NVIDIA stock" END=2026-03-01 \
  bash scripts/run_stock_full_pipeline.sh data
```

## Expected raw/intermediate layout

```text
data/raw/stock_data/NVDA/
data/raw/macro_stock_data/SPY/
data/raw/macro_stock_data/SOXX/
data/raw/macro_stock_data/IEF/
data/raw/news_headlines_nvda/
data/raw/trends_nvda_pytrends/
data/interim/nvda_trends_daily_consistent.csv
data/processed/nvda_trends_processed.csv
data/processed/nvda_news_daily_sentiment.csv
```

## Main generated outputs

```text
data/model_feed/nvda_model_dataset.csv
data/model_feed/nvda_model_dataset_clean.csv
data/model_feed/nvda_model_dataset_audit.xlsx
artifacts/models/nvda_random_search_reduced_features/
artifacts/models/nvda_walk_forward_random_search_bestparams/
artifacts/reports/nvda_baseline_models_linear_svm_ablations/
artifacts/reports/nvda_model_comparison/
artifacts/reports/nvda_full_pipeline_summary.csv
```

## Final model features

The cleaned modelling dataset contains:

```text
target_direction
target_next_return
log_return
overnight_return
momentum_5d
momentum_20d
volatility_20d
volume_change
volume_20d_avg
avg_sentiment
sentiment_std
news_count
spy_return
soxx_return
ief_return
trends_zscore_30d
trends_momentum_7d
trends_spike
```

`target_next_return` is required for trading-oriented performance metrics such as strategy Sharpe, equity curves and drawdown.

## Notes for reviewers

The repository can be reviewed at two levels:

1. Start from `<symbol>_model_dataset_clean.csv` and reproduce the empirical tables and figures.
2. Run the generic pipeline from raw/API checkpoints through final model comparison.

The first route is the most convenient for thesis review. The second route documents the full data-construction process.
