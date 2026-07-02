# Dataset schema

The final modelling dataset is expected at:

```text
data/model_feed/model_dataset_clean.csv
```

It is a chronological daily dataset used by both the classical benchmarks and the LSTM walk-forward evaluation.

## Required columns

| Column | Type | Description |
|---|---|---|
| `date` | date | Trading date. |
| `target_next_return` | float | Next-day NVDA log return; used for trading-oriented metrics. |
| `target_direction` | integer | Direction target: `1` for positive next-day return, `0` otherwise. |
| `log_return` | float | Current-day NVDA log return. |
| `overnight_return` | float | Return from previous close to current open. |
| `momentum_5d` | float | Five-day rolling mean of NVDA log returns. |
| `momentum_20d` | float | Twenty-day rolling mean of NVDA log returns. |
| `volatility_20d` | float | Twenty-day rolling standard deviation of NVDA log returns. |
| `volume_change` | float | Percentage change in daily volume. |
| `volume_20d_avg` | float | Twenty-day rolling average of volume. |
| `spy_return` | float | SPY log return as broad market proxy. |
| `soxx_return` | float | SOXX log return as semiconductor-sector proxy. |
| `ief_return` | float | IEF log return as interest-rate/bond proxy. |
| `avg_sentiment` | float | Daily average VADER compound sentiment from news headlines. |
| `sentiment_std` | float | Daily sentiment dispersion. |
| `news_count` | integer | Number of aligned news items for the trading date. |
| `trends_zscore_30d` | float | Thirty-day Google Trends abnormal attention z-score. |
| `trends_momentum_7d` | float | Seven-day momentum in Google Trends interest. |
| `trends_spike` | integer | Indicator for unusually high Google Trends interest. |

## Validation checks

Run:

```cmd
thesis-preprocess validate --input data\model_feed\model_dataset_clean.csv
thesis-preprocess audit --input data\model_feed\model_dataset_clean.csv --output data\model_feed\model_dataset_audit.xlsx
```

The validation step checks basic dataset shape, date range, duplicate dates, large date gaps, missing values, zero-variance columns and target correlations.

## Notes

- Rows must remain chronologically ordered.
- The evaluation pipeline uses strictly chronological train/validation/test splits.
- `target_direction` is the classification target.
- `target_next_return` is not a model feature; it is used to evaluate trading-oriented performance.
