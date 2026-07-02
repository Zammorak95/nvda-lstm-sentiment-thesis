# Preprocessing pipeline

The final modelling dataset is produced by a preprocessing pipeline that combines market data, news sentiment and Google Trends attention features.

The canonical command is:

```cmd
thesis-preprocess --help
```

The canonical source file is:

```text
src/thesis/preprocessing/data_pipeline.py
```

This replaces the older loose scripts such as `data_pipeline_combined.py`, `Stock_preprocessing.py`, `daily google trends series.py` and `model_dataset_clean_and_analyze.py` as the documented entry point. The older files are useful historical references, but the repository should point users to `thesis-preprocess`.

## Output target

The final clean modelling dataset used by all model-evaluation scripts is:

```text
data/model_feed/model_dataset_clean.csv
```

## Required raw/intermediate inputs

The default raw-data layout is:

```text
data/raw/news_headlines/                  Monthly NVDA news CSV files
data/raw/stock_data/NVDA/                 Raw NVDA EOD CSV
data/raw/macro_stock_data/SPY/            Raw SPY EOD CSV
data/raw/macro_stock_data/SOXX/           Raw SOXX EOD CSV
data/raw/macro_stock_data/IEF/            Raw IEF EOD CSV
data/raw/trends/                          Google Trends exports
```

The Google Trends folder should contain:

```text
multiTimeline.csv                         Monthly reference series
multiTimeline(1).csv, multiTimeline(2).csv, ... daily chunks
```

## Pipeline stages

### 1. Combine raw news files

```cmd
thesis-preprocess news-combine
```

Output:

```text
data/interim/news_headlines_master.csv
```

### 2. Clean and trading-align news

```cmd
thesis-preprocess news-clean
```

This removes unusable rows, parses timestamps and aligns news to U.S. trading days. News after the market close is assigned to the next trading day.

Output:

```text
data/processed/news_headlines_clean.csv
```

### 3. Build daily VADER sentiment features

```cmd
thesis-preprocess news-sentiment
```

Output:

```text
data/processed/news_daily_sentiment.csv
```

Main features:

```text
avg_sentiment
sentiment_std
news_count
```

### 4. Process stock and ETF EOD data

```cmd
thesis-preprocess stock-clean-all
```

This creates return, momentum, volatility and volume features for NVDA and log-return features for SPY, SOXX and IEF.

Outputs:

```text
data/processed/NVDA_eod_processed.csv
data/processed/SPY_eod_processed.csv
data/processed/SOXX_eod_processed.csv
data/processed/IEF_eod_processed.csv
```

### 5. Reconstruct daily Google Trends series

```cmd
thesis-preprocess trends-reconstruct
```

This uses the monthly Google Trends export as a reference anchor and rescales daily chunks into a consistent daily series.

Output:

```text
data/interim/nvidia_trends_daily_consistent.csv
```

### 6. Build Google Trends attention features

```cmd
thesis-preprocess trends-clean
```

Output:

```text
data/processed/nvidia_trends_processed.csv
```

Main features:

```text
trends_zscore_30d
trends_momentum_7d
trends_spike
```

### 7. Merge the model dataset

```cmd
thesis-preprocess build-model
```

Output:

```text
data/model_feed/model_dataset.csv
```

### 8. Write the compact clean dataset

```cmd
thesis-preprocess write-clean
```

Output:

```text
data/model_feed/model_dataset_clean.csv
```

### 9. Validate and audit the final dataset

```cmd
thesis-preprocess validate --input data/model_feed/model_dataset_clean.csv
thesis-preprocess audit --input data/model_feed/model_dataset_clean.csv --output data/model_feed/model_dataset_audit.xlsx
```

## Full preprocessing run

If all raw files are in the default locations, run:

```cmd
thesis-preprocess all
```

This runs all stages and writes:

```text
data/model_feed/model_dataset.csv
data/model_feed/model_dataset_clean.csv
data/model_feed/model_dataset_audit.xlsx
```

## Custom path examples

Clean one stock file:

```cmd
thesis-preprocess stock-clean ^
  --symbol NVDA ^
  --input data\raw\stock_data\NVDA\NVDA_eod_chunked_2019-03-01_to_2026-03-01.csv ^
  --output data\processed\NVDA_eod_processed.csv
```

Use a different base data directory:

```cmd
thesis-preprocess --base-dir D:\thesis_data all
```

## How this connects to the thesis method

The preprocessing pipeline implements the data work described in the Materials and Methods chapter:

- stock data are cleaned chronologically and transformed into returns, momentum, volatility and volume features;
- news items are aligned to trading days and converted into daily sentiment features;
- Google Trends chunks are reconstructed into a consistent daily attention series;
- processed stock, macro, sentiment and attention variables are merged by trading date;
- the final feature matrix is validated before model training.

## Important limitation

Full raw-data reproduction requires access to the original raw CSV files, API outputs and Google Trends exports. If those files cannot be redistributed, the recommended reproducibility anchor is the cleaned dataset:

```text
data/model_feed/model_dataset_clean.csv
```
