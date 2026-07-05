# Raw-data acquisition

This page documents how the raw market, news and Google Trends data are obtained before preprocessing.

The active generic entry point is:

```bash
SYMBOL=NVDA KEYWORD="NVIDIA stock" END=2026-03-01 \
  bash scripts/run_stock_full_pipeline.sh data
```

## StockData.org market data

The data phase downloads EOD OHLCV data for the target ticker and macro/market proxies:

```text
Target ticker: SYMBOL, e.g. NVDA or AMD
Market proxy: SPY
Sector proxy: SOXX
Bond/rate proxy: IEF
```

If raw or processed files already exist, they are reused unless `FORCE=1` is set.

## StockData.org news headlines

The pipeline uses StockData.org symbol-filtered news headlines. When `NEWS_START=auto`, the pipeline first tries to infer the earliest local news date. If no local news files exist, it performs an API-conscious hierarchical scan:

```text
year probes -> month probes -> day probes
```

After the news start is resolved, news is fetched day by day and written into monthly checkpoint CSV files. The default per-day article limit is:

```bash
NEWS_LIMIT_PER_DAY=25
```

## Google Trends through PyTrends

Google Trends is now collected through PyTrends inside the generic pipeline.

The process is:

1. Fetch one full-period reference series for `KEYWORD`.
2. Fetch smaller daily chunks over the same period.
3. Scale the daily chunks to the full-period reference.
4. Chain-link adjacent chunks to reduce artificial jumps.
5. Interpolate to a continuous daily series.
6. Build attention features such as z-score, 7-day momentum and spike indicators.

The main command is:

```bash
SYMBOL=NVDA KEYWORD="NVIDIA stock" END=2026-03-01 \
  bash scripts/run_stock_full_pipeline.sh data
```

Useful Google Trends settings:

```text
KEYWORD                 Google Trends search term
trends_geo              Empty by default; set in code/CLI if needed
trends_gprop            Empty by default; web search interest
trends_chunk_days       90 days by default
trends_sleep_min/max    Polite waiting between requests
```

The implementation is intentionally conservative: it uses retries, checkpoint files and polite waits, but it does not rotate proxies or fake identities.

## Raw-to-clean sequence

Full data construction route:

```bash
SYMBOL=NVDA KEYWORD="NVIDIA stock" END=2026-03-01 bash scripts/run_stock_full_pipeline.sh data
SYMBOL=NVDA KEYWORD="NVIDIA stock" bash scripts/run_stock_full_pipeline.sh reduced
```

This produces:

```text
data/model_feed/nvda_model_dataset.csv
data/model_feed/nvda_model_dataset_clean.csv
data/model_feed/nvda_model_dataset_audit.xlsx
```

For AMD, replace `SYMBOL=NVDA` and `KEYWORD="NVIDIA stock"` with the AMD values.

## Practical notes

- Raw generated files are checkpointed and reused by default.
- Use `FORCE=1` only when you intentionally want to refetch or rebuild.
- Google Trends values can depend on keyword, geography, time window and collection timing.
- Keep local credentials in `.env`; use `.env.example` as the template.
