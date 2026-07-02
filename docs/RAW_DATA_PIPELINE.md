# Raw-data pipeline notes

The most reliable reproduction path starts from the cleaned modelling dataset:

```text
data/model_feed/model_dataset_clean.csv
```

A full raw-data rebuild is possible in principle, but it depends on external API access, source-data licences and local raw files. Therefore the official thesis reproduction workflow treats `model_dataset_clean.csv` as the minimum reproducibility input.

## Conceptual raw-data flow

```text
1. Market data
   StockData.org / market data source
   -> raw OHLCV files
   -> processed daily return and volume features

2. News data
   raw financial-news headline files
   -> cleaned and trading-day aligned headline data
   -> VADER sentiment features

3. Google Trends data
   exported monthly/daily Google Trends files
   -> normalized daily attention series
   -> z-score, momentum and spike features

4. Merge step
   stock features + sentiment features + trends features + macro/ETF features
   -> data/model_feed/model_dataset.csv
   -> curated clean feature set
   -> data/model_feed/model_dataset_clean.csv
```

## Final model features

The cleaned modelling dataset is expected to contain the target and a curated feature set similar to:

```text
target_direction
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

`target_next_return` is optional but required for trading-oriented performance metrics such as strategy Sharpe, equity curves and drawdown.

## Why the clean dataset is the reproducibility anchor

The clean dataset is the best reproducibility anchor because:

- raw data sources may require API keys;
- some source data may not be legally redistributable;
- Google Trends exports can depend on export timing and scaling;
- raw news files may have licensing restrictions;
- exact raw downloads may be difficult to recreate later.

For this reason, the codebase is organized so that the empirical results can be reproduced from `model_dataset_clean.csv` even when raw-source access is unavailable.

## Recommended reporting language

Use language like:

> The empirical workflow is reproducible from the cleaned modelling dataset, which contains the final synchronized feature matrix used for all model evaluations. Full raw-data reconstruction requires access to the original market, news and Google Trends sources and may therefore depend on data-source availability and licensing.

## If publishing the clean dataset

If the clean dataset is legally redistributable and small enough, either:

1. commit it deliberately with `git add -f data/model_feed/model_dataset_clean.csv`, or
2. publish it as a GitHub Release asset and instruct users to download it into `data/model_feed/`.

Do not accidentally commit raw data, API keys, news full text, or local machine paths.
