# Raw-data acquisition

This page documents how the raw market and news data were obtained before preprocessing.

The canonical command is:

```cmd
thesis-fetch-stockdata --help
```

The canonical source file is:

```text
src/thesis/data_acquisition/stockdata_api.py
```

It replaces the older loose script `nvda_stockdata_fetch_combined.py` as the documented entry point for StockData.org downloads.

## Authentication

Create a local `.env` file or set an environment variable manually. Do not commit API keys.

Windows CMD:

```cmd
set STOCKDATA_API_TOKEN=your_token_here
```

PowerShell:

```powershell
$env:STOCKDATA_API_TOKEN="your_token_here"
```

The command also accepts `STOCKDATA_API_KEY`.

## Market data from StockData.org

### NVDA EOD data

```cmd
thesis-fetch-stockdata market ^
  --mode eod ^
  --symbol NVDA ^
  --start 2019-03-01 ^
  --end 2026-03-01 ^
  --csv
```

Default output:

```text
data/raw/stock_data/NVDA/NVDA_eod_2019-03-01_to_2026-03-01.parquet
data/raw/stock_data/NVDA/NVDA_eod_2019-03-01_to_2026-03-01.csv
```

### Macro/market proxy EOD data

```cmd
thesis-fetch-stockdata market --mode eod --symbol SPY  --start 2019-03-01 --end 2026-03-01 --csv --outdir data\raw\macro_stock_data\SPY
thesis-fetch-stockdata market --mode eod --symbol SOXX --start 2019-03-01 --end 2026-03-01 --csv --outdir data\raw\macro_stock_data\SOXX
thesis-fetch-stockdata market --mode eod --symbol IEF  --start 2019-03-01 --end 2026-03-01 --csv --outdir data\raw\macro_stock_data\IEF
```

### Intraday data

The thesis later uses a daily modelling dataset, but intraday data can be downloaded if needed:

```cmd
thesis-fetch-stockdata market ^
  --mode intraday ^
  --symbol NVDA ^
  --start 2024-01-01 ^
  --end 2024-02-01 ^
  --interval minute ^
  --chunk-days 7 ^
  --csv
```

## News headlines from StockData.org

```cmd
thesis-fetch-stockdata news ^
  --symbols NVDA ^
  --start 2019-03-01 ^
  --end 2026-03-01 ^
  --chunk-days 30 ^
  --csv
```

Default output:

```text
data/raw/news_headlines/NVDA_news_2019-03-01_to_2026-03-01.parquet
data/raw/news_headlines/NVDA_news_2019-03-01_to_2026-03-01.csv
```

The news command has a configurable endpoint:

```cmd
thesis-fetch-stockdata news --url <endpoint> --param key=value
```

Use this if StockData.org changes the news endpoint or requires additional query parameters for your subscription.

## Google Trends data

Google Trends data were collected manually because Google Trends does not provide a stable official bulk-download API for this exact workflow.

Recommended manual collection process:

1. Go to Google Trends.
2. Search for `Nvidia` or the exact query used in the thesis.
3. Set geography and category consistently. Document the choice, for example worldwide search interest.
4. Download one full-period monthly series. Save it as:

```text
data/raw/trends/multiTimeline.csv
```

5. Download smaller overlapping daily windows. Google Trends gives daily resolution for shorter time windows. Save them as:

```text
data/raw/trends/multiTimeline(1).csv
data/raw/trends/multiTimeline(2).csv
data/raw/trends/multiTimeline(3).csv
...
```

6. Run:

```cmd
thesis-preprocess trends-reconstruct
thesis-preprocess trends-clean
```

The preprocessing step uses the monthly series as a global scale anchor and rescales the daily chunks into a consistent daily attention series.

## Raw-to-clean sequence

After all raw files are present:

```cmd
thesis-preprocess all
```

This produces:

```text
data/model_feed/model_dataset.csv
data/model_feed/model_dataset_clean.csv
data/model_feed/model_dataset_audit.xlsx
```

## Important notes

- API keys must never be committed to GitHub.
- Raw data are ignored by Git by default.
- Raw news and market data may be subject to licensing restrictions.
- Google Trends exports should be documented carefully because export timing and query settings can affect the resulting series.
