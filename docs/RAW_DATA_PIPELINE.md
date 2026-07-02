# Raw-data pipeline notes

The most reliable reproduction path starts from the cleaned modelling dataset:

```text
data/model_feed/model_dataset_clean.csv
```

A full raw-data rebuild is possible in principle, but it depends on external API access, source-data licences and local raw files. Therefore the official thesis reproduction workflow treats `model_dataset_clean.csv` as the minimum reproducibility input when raw-source access is unavailable.

Canonical commands:

```cmd
thesis-fetch-stockdata --help
thesis-preprocess --help
```

## Conceptual raw-data flow

```text
1. StockData.org market data
   thesis-fetch-stockdata market
   -> raw NVDA/SPY/SOXX/IEF OHLCV files
   -> processed daily return, momentum, volatility and volume features

2. StockData.org news headlines
   thesis-fetch-stockdata news
   -> raw financial-news headline files
   -> cleaned and trading-day aligned headline data
   -> VADER sentiment features

3. Google Trends data
   manual Google Trends exports
   -> full-period monthly anchor export
   -> shorter daily-window exports
   -> normalized daily attention series
   -> z-score, momentum and spike features

4. Merge step
   stock features + sentiment features + trends features + macro/ETF features
   -> data/model_feed/model_dataset.csv
   -> curated clean feature set
   -> data/model_feed/model_dataset_clean.csv
```

## StockData.org raw acquisition

Set a local environment variable first. Never commit the token.

```cmd
thesis-fetch-stockdata market --mode eod --symbol NVDA --start 2019-03-01 --end 2026-03-01 --csv
thesis-fetch-stockdata market --mode eod --symbol SPY  --start 2019-03-01 --end 2026-03-01 --csv --outdir data\raw\macro_stock_data\SPY
thesis-fetch-stockdata market --mode eod --symbol SOXX --start 2019-03-01 --end 2026-03-01 --csv --outdir data\raw\macro_stock_data\SOXX
thesis-fetch-stockdata market --mode eod --symbol IEF  --start 2019-03-01 --end 2026-03-01 --csv --outdir data\raw\macro_stock_data\IEF
thesis-fetch-stockdata news --symbols NVDA --start 2019-03-01 --end 2026-03-01 --chunk-days 30 --csv
```

## Google Trends manual acquisition

Google Trends was collected manually. The reproducible description is:

1. Open Google Trends.
2. Use the same query term and settings as the thesis, for example `Nvidia`, same geography, category and search type.
3. Download one full-period monthly overview and save it as `data/raw/trends/multiTimeline.csv`.
4. Download smaller overlapping windows so that Google Trends provides daily resolution. Save them as `multiTimeline(1).csv`, `multiTimeline(2).csv`, etc.
5. Run `thesis-preprocess trends-reconstruct` and `thesis-preprocess trends-clean`.

The reconstruction step uses the monthly overview as a scale anchor and then rescales the daily chunks into one continuous daily attention series.

## Expected raw-data layout

```text
data/raw/news_headlines/                  Monthly NVDA news CSV files
data/raw/stock_data/NVDA/                 Raw NVDA EOD CSV
data/raw/macro_stock_data/SPY/            Raw SPY EOD CSV
data/raw/macro_stock_data/SOXX/           Raw SOXX EOD CSV
data/raw/macro_stock_data/IEF/            Raw IEF EOD CSV
data/raw/trends/                          Google Trends monthly and daily exports
```

## Main preprocessing outputs

```text
data/interim/news_headlines_master.csv
data/processed/news_headlines_clean.csv
data/processed/news_daily_sentiment.csv
data/processed/NVDA_eod_processed.csv
data/processed/SPY_eod_processed.csv
data/processed/SOXX_eod_processed.csv
data/processed/IEF_eod_processed.csv
data/interim/nvidia_trends_daily_consistent.csv
data/processed/nvidia_trends_processed.csv
data/model_feed/model_dataset.csv
data/model_feed/model_dataset_clean.csv
data/model_feed/model_dataset_audit.xlsx
```

## Final model features

The cleaned modelling dataset is expected to contain the target and a curated feature set similar to:

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

> The empirical workflow is reproducible from the cleaned modelling dataset, which contains the final synchronized feature matrix used for all model evaluations. Full raw-data reconstruction requires access to the original market and news API data and manually exported Google Trends files and may therefore depend on data-source availability and licensing.

## If publishing the clean dataset

If the clean dataset is legally redistributable and small enough, either:

1. commit it deliberately with `git add -f data/model_feed/model_dataset_clean.csv`, or
2. publish it as a GitHub Release asset and instruct users to download it into `data/model_feed/`.

Do not accidentally commit raw data, API keys, news full text, or local machine paths.
