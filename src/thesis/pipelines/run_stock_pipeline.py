#!/usr/bin/env python3
"""
Generic thesis data pipeline for a target equity symbol.

This orchestrates the existing thesis scripts while adding resumable collection for
StockData news and Google Trends. It is intentionally conservative with external
services: it uses checkpoint files, retries, backoff, and randomized polite waits,
but it does not rotate proxies, identities, or fake browser sessions.

Examples:
  python -u run_stock_pipeline.py all --symbol AMD --keyword "AMD stock" --start 2022-01-01 --end 2026-02-26
  python -u run_stock_pipeline.py fetch-news --symbol AMD --news-start 2022-01-01 --news-end 2026-02-26
  python -u run_stock_pipeline.py fetch-trends --symbol AMD --keyword "AMD stock" --start 2016-01-01 --end 2026-03-01
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests


STOCKDATA_NEWS_URL = "https://api.stockdata.org/v1/news/all"


# -----------------------------------------------------------------------------
# Paths and small utilities
# -----------------------------------------------------------------------------
def project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in (current.parent, *current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return current.parents[4]


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def date_range(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def month_start_end(d: date) -> tuple[date, date]:
    start = d.replace(day=1)
    if d.month == 12:
        next_month = d.replace(year=d.year + 1, month=1, day=1)
    else:
        next_month = d.replace(month=d.month + 1, day=1)
    return start, next_month - timedelta(days=1)


def month_iter(start: date, end: date) -> Iterable[tuple[date, date]]:
    cur = start.replace(day=1)
    while cur <= end:
        m_start, m_end = month_start_end(cur)
        yield max(m_start, start), min(m_end, end)
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1, day=1)
        else:
            cur = cur.replace(month=cur.month + 1, day=1)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def random_sleep(min_seconds: float, max_seconds: float) -> None:
    if max_seconds <= 0:
        return
    lo = max(0.0, min_seconds)
    hi = max(lo, max_seconds)
    time.sleep(random.uniform(lo, hi))


def run_cmd(cmd: list[str], dry_run: bool = False) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def get_stockdata_token() -> str:
    for name in ("STOCKDATA_API_TOKEN", "STOCKDATA_API_KEY"):
        token = os.getenv(name)
        if token:
            return token
    raise RuntimeError("Missing StockData token. Export STOCKDATA_API_TOKEN or STOCKDATA_API_KEY, e.g. set -a; source .env; set +a")


def script_paths(root: Path) -> tuple[Path, Path]:
    stock_fetcher = root / "src/thesis/One doc/nvda_stockdata_fetch_combined.py"
    combined_pipeline = root / "src/thesis/One doc/data_pipeline_combined.py"
    if not stock_fetcher.exists():
        raise FileNotFoundError(f"Stock fetcher not found: {stock_fetcher}")
    if not combined_pipeline.exists():
        raise FileNotFoundError(f"Combined pipeline not found: {combined_pipeline}")
    return stock_fetcher, combined_pipeline


def symbol_lower(symbol: str) -> str:
    return symbol.strip().lower()


def raw_stock_path(data_dir: Path, symbol: str, start: str, end: str, macro: bool = False) -> Path:
    base = data_dir / "raw" / ("macro_stock_data" if macro else "stock_data") / symbol.upper()
    return base / f"{symbol.upper()}_eod_{start}_to_{end}.csv"


def processed_stock_path(data_dir: Path, symbol: str) -> Path:
    return data_dir / "processed" / f"{symbol.upper()}_eod_processed.csv"


# -----------------------------------------------------------------------------
# StockData EOD orchestration
# -----------------------------------------------------------------------------
def fetch_eod(args: argparse.Namespace, symbol: str, macro: bool = False) -> Path:
    root = args.root
    data_dir = args.data_dir
    stock_fetcher, _ = script_paths(root)
    outdir = data_dir / "raw" / ("macro_stock_data" if macro else "stock_data") / symbol.upper()
    outdir.mkdir(parents=True, exist_ok=True)
    expected = raw_stock_path(data_dir, symbol, args.start, args.end, macro=macro)

    if expected.exists() and not args.force:
        print(f"[SKIP] EOD raw exists for {symbol}: {expected}")
        return expected

    run_cmd(
        [
            sys.executable,
            str(stock_fetcher),
            "--mode",
            "eod",
            "--symbol",
            symbol.upper(),
            "--start",
            args.start,
            "--end",
            args.end,
            "--outdir",
            str(outdir),
            "--csv",
            "--continue_on_empty",
        ],
        dry_run=args.dry_run,
    )
    return expected


def clean_eod(args: argparse.Namespace, symbol: str, macro: bool = False) -> Path:
    root = args.root
    data_dir = args.data_dir
    _, combined_pipeline = script_paths(root)
    input_path = raw_stock_path(data_dir, symbol, args.start, args.end, macro=macro)
    output_path = processed_stock_path(data_dir, symbol)

    if not input_path.exists() and not args.dry_run:
        raise FileNotFoundError(f"Missing raw EOD input for {symbol}: {input_path}")

    # data_pipeline_combined only accepts stock-clean symbols from the original default list.
    pipeline_symbol = symbol.upper() if symbol.upper() in {"NVDA", "SPY", "SOXX", "IEF"} else "NVDA"
    run_cmd(
        [
            sys.executable,
            str(combined_pipeline),
            "stock-clean",
            "--symbol",
            pipeline_symbol,
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        dry_run=args.dry_run,
    )
    return output_path


# -----------------------------------------------------------------------------
# Google Trends: resumable, polite, chunked fetch
# -----------------------------------------------------------------------------
def fetch_trends_interest(pytrends: Any, keyword: str, start: str, end: str, geo: str, gprop: str) -> pd.DataFrame:
    timeframe = f"{start} {end}"
    pytrends.build_payload(kw_list=[keyword], cat=0, timeframe=timeframe, geo=geo, gprop=gprop)
    df = pytrends.interest_over_time()
    if df.empty:
        raise RuntimeError(f"No Google Trends data returned for {timeframe}")
    df = df.reset_index()
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    df = df.rename(columns={keyword: "value"})
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["date", "value"]].dropna().sort_values("date").reset_index(drop=True)


def date_chunks(start_date: str, end_date: str, chunk_days: int) -> Iterable[tuple[str, str]]:
    start = parse_date(start_date)
    end = parse_date(end_date)
    cur = start
    while cur <= end:
        chunk_end = min(end, cur + timedelta(days=chunk_days - 1))
        yield cur.isoformat(), chunk_end.isoformat()
        cur = chunk_end + timedelta(days=1)


def compute_scale(chunk: pd.DataFrame, reference: pd.DataFrame) -> float:
    tmp = chunk.copy()
    ref = reference.copy()
    tmp["month"] = tmp["date"].values.astype("datetime64[M]")
    ref["month"] = ref["date"].values.astype("datetime64[M]")
    chunk_monthly = tmp.groupby("month", as_index=False)["value"].mean()
    ref_monthly = ref.groupby("month", as_index=False)["value"].mean().rename(columns={"value": "ref_value"})
    merged = chunk_monthly.merge(ref_monthly, on="month", how="inner")
    if merged.empty:
        return 1.0
    ratios = merged["ref_value"] / merged["value"].replace(0, np.nan)
    ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()
    return float(ratios.median()) if not ratios.empty else 1.0


def fetch_with_retries(fetch_fn, retries: int, min_wait: float, max_wait: float):
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fetch_fn()
        except Exception as exc:  # noqa: BLE001 - keep checkpointing on external-service failures
            last_exc = exc
            if attempt >= retries:
                break
            wait = random.uniform(min_wait, max_wait) * attempt
            print(f"[WARN] attempt {attempt}/{retries} failed: {exc}. Sleeping {wait:.1f}s before retry.", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"All retries failed: {last_exc}")


def fetch_google_trends(args: argparse.Namespace) -> Path:
    from pytrends.request import TrendReq  # noqa: PLC0415

    symbol = args.symbol.upper()
    slug = symbol_lower(symbol)
    keyword = args.keyword or f"{symbol} stock"
    raw_dir = args.data_dir / "raw" / f"trends_{slug}_pytrends"
    interim_path = args.data_dir / "interim" / f"{slug}_trends_daily_consistent.csv"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent(interim_path)

    manifest_path = raw_dir / f"{slug}_trends_fetch_manifest.csv"
    reference_path = raw_dir / f"{slug}_reference_full_period.csv"

    if interim_path.exists() and not args.force:
        print(f"[SKIP] Trends interim exists: {interim_path}")
        return interim_path

    pytrends = TrendReq(
        hl=args.trends_hl,
        tz=args.trends_tz,
        timeout=(10, 30),
        retries=0,
        backoff_factor=0,
    )

    if reference_path.exists() and not args.force:
        print(f"[RESUME] Loading reference: {reference_path}")
        reference = pd.read_csv(reference_path)
        reference["date"] = pd.to_datetime(reference["date"])
    else:
        print(f"[TRENDS] Fetching reference for '{keyword}' {args.start} -> {args.end}")
        reference = fetch_with_retries(
            lambda: fetch_trends_interest(pytrends, keyword, args.start, args.end, args.trends_geo, args.trends_gprop),
            args.trends_retries,
            args.trends_sleep_min,
            args.trends_sleep_max,
        )
        reference.to_csv(reference_path, index=False)
        random_sleep(args.trends_sleep_min, args.trends_sleep_max)

    chunks: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []

    if manifest_path.exists() and not args.force:
        try:
            manifest_rows = pd.read_csv(manifest_path).to_dict("records")
        except Exception:
            manifest_rows = []

    for i, (start, end) in enumerate(date_chunks(args.start, args.end, args.trends_chunk_days), start=1):
        chunk_path = raw_dir / f"{slug}_chunk_{i:03d}_{start}_to_{end}.csv"
        print(f"[TRENDS] chunk {i:03d}: {start} -> {end}")

        if chunk_path.exists() and not args.force:
            chunk = pd.read_csv(chunk_path)
            chunk["date"] = pd.to_datetime(chunk["date"])
            print(f"  [RESUME] loaded {len(chunk):,} rows from {chunk_path.name}")
        else:
            chunk = fetch_with_retries(
                lambda s=start, e=end: fetch_trends_interest(pytrends, keyword, s, e, args.trends_geo, args.trends_gprop),
                args.trends_retries,
                args.trends_sleep_min,
                args.trends_sleep_max,
            )
            chunk.to_csv(chunk_path, index=False)
            manifest_rows.append(
                {
                    "chunk": i,
                    "start": start,
                    "end": end,
                    "rows": len(chunk),
                    "status": "done",
                    "saved_path": str(chunk_path),
                    "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            pd.DataFrame(manifest_rows).drop_duplicates(subset=["chunk", "start", "end"], keep="last").to_csv(
                manifest_path, index=False
            )
            random_sleep(args.trends_sleep_min, args.trends_sleep_max)

        scale = compute_scale(chunk, reference)
        chunk = chunk.copy()
        chunk["value"] = chunk["value"] * scale
        chunk["chunk"] = i
        chunk["scale_to_reference"] = scale
        chunks.append(chunk)

    if not chunks:
        raise RuntimeError("No Google Trends chunks found or fetched.")

    scaled_chunks = chunks
    for i in range(1, len(scaled_chunks)):
        prev = scaled_chunks[i - 1]
        curr = scaled_chunks[i]
        prev_last = float(prev.loc[prev["date"] == prev["date"].max(), "value"].iloc[0])
        curr_first = float(curr.loc[curr["date"] == curr["date"].min(), "value"].iloc[0])
        factor = prev_last / curr_first if abs(curr_first) > 1e-9 else 1.0
        scaled_chunks[i] = curr.copy()
        scaled_chunks[i]["value"] = scaled_chunks[i]["value"] * factor
        scaled_chunks[i]["chain_link_factor"] = factor

    daily = pd.concat(scaled_chunks, ignore_index=True).sort_values("date")
    daily = daily.groupby("date", as_index=False)["value"].mean()
    daily = daily.set_index("date").asfreq("D")
    daily["value"] = daily["value"].interpolate(limit_direction="both")
    daily = daily.reset_index().rename(columns={"value": f"{slug}_trends"})
    daily.to_csv(interim_path, index=False)

    print(f"[TRENDS] Saved {len(daily):,} rows -> {interim_path}")
    return interim_path


def clean_trends(args: argparse.Namespace) -> Path:
    _, combined_pipeline = script_paths(args.root)
    slug = symbol_lower(args.symbol)
    input_path = args.data_dir / "interim" / f"{slug}_trends_daily_consistent.csv"
    output_path = args.data_dir / "processed" / f"{slug}_trends_processed.csv"
    run_cmd(
        [
            sys.executable,
            str(combined_pipeline),
            "trends-clean",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        dry_run=args.dry_run,
    )
    return output_path


# -----------------------------------------------------------------------------
# StockData news: resumable daily fetch, monthly files, progress metadata
# -----------------------------------------------------------------------------
def news_month_path(args: argparse.Namespace, year: int, month: int) -> Path:
    slug = symbol_lower(args.symbol)
    return args.data_dir / "raw" / f"news_headlines_{slug}" / f"{args.symbol.upper()}_news_{year:04d}_{month:02d}.csv"


def read_existing_progress(progress_path: Path) -> set[str]:
    if not progress_path.exists():
        return set()
    try:
        df = pd.read_csv(progress_path)
    except Exception:
        return set()
    if "status" not in df.columns or "api_day" not in df.columns:
        return set()
    return set(df.loc[df["status"].eq("done"), "api_day"].astype(str))


def append_progress(progress_path: Path, row: dict[str, Any]) -> None:
    ensure_parent(progress_path)
    exists = progress_path.exists()
    with progress_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def flatten_article(article: dict[str, Any], symbol: str, api_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol_target": symbol.upper(),
        "uuid": article.get("uuid", ""),
        "published_at": article.get("published_at", ""),
        "title": article.get("title", ""),
        "description": article.get("description", ""),
        "keywords": article.get("keywords", ""),
        "snippet": article.get("snippet", ""),
        "url": article.get("url", ""),
        "source": article.get("source") or article.get("domain") or article.get("source_name") or "",
        "image_url": article.get("image_url") or article.get("imageUrl") or "",
        "entities_json": json.dumps(article.get("entities", []), ensure_ascii=False),
        "raw_json": json.dumps(article, ensure_ascii=False),
        "api_day": api_meta["api_day"],
        "api_found": api_meta["api_found"],
        "api_returned": api_meta["api_returned"],
        "api_limit_per_day": api_meta["api_limit_per_day"],
        "api_query_mode": api_meta["api_query_mode"],
        "fetch_timestamp_utc": api_meta["fetch_timestamp_utc"],
    }


def write_month_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    df_new = pd.DataFrame(rows)
    if path.exists():
        df_old = pd.read_csv(path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    if "uuid" in df.columns:
        df = df.drop_duplicates(subset=["uuid"], keep="first")
    else:
        df = df.drop_duplicates()
    df.to_csv(path, index=False)


def fetch_news_day(session: requests.Session, token: str, symbol: str, day: date, limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    params = {
        "api_token": token,
        "symbols": symbol.upper(),
        "filter_entities": "true",
        "language": "en",
        "published_on": day.isoformat(),
        "limit": str(limit),
        "page": "1",
    }
    response = session.get(STOCKDATA_NEWS_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(f"StockData error: {payload['error']}")
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    articles = payload.get("data", []) if isinstance(payload, dict) else []
    return articles[:limit], meta


def fetch_news(args: argparse.Namespace) -> Path:
    token = get_stockdata_token()
    symbol = args.symbol.upper()
    slug = symbol_lower(symbol)
    raw_dir = args.data_dir / "raw" / f"news_headlines_{slug}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    progress_path = raw_dir / f"{symbol}_news_fetch_progress.csv"
    completed_days = read_existing_progress(progress_path) if not args.force else set()

    start = parse_date(args.news_start or args.start)
    end = parse_date(args.news_end or args.end)
    session = requests.Session()
    session.headers.update({"User-Agent": "thesis-stock-news-fetcher/1.0"})

    for m_start, m_end in month_iter(start, end):
        month_path = news_month_path(args, m_start.year, m_start.month)
        print(f"[NEWS] {symbol} {m_start:%Y-%m}: {m_start} -> {m_end}", flush=True)
        nonzero_days = 0
        month_new_rows = 0

        for day in date_range(m_start, m_end):
            if day.isoformat() in completed_days and not args.force:
                print(f"  [SKIP] {day} already completed")
                continue

            try:
                articles, meta = fetch_with_retries(
                    lambda d=day: fetch_news_day(session, token, symbol, d, args.news_limit_per_day),
                    args.news_retries,
                    args.news_sleep_min,
                    args.news_sleep_max,
                )
                found = int(meta.get("found", 0) or 0)
                returned = len(articles)
                print(f"  [DAY] {day} found={found} returned={returned}", flush=True)

                fetch_ts = datetime.now(timezone.utc).isoformat()
                api_meta = {
                    "api_day": day.isoformat(),
                    "api_found": found,
                    "api_returned": returned,
                    "api_limit_per_day": args.news_limit_per_day,
                    "api_query_mode": "published_on+filter_entities",
                    "fetch_timestamp_utc": fetch_ts,
                }
                rows = [flatten_article(a, symbol, api_meta) for a in articles]
                if rows:
                    write_month_rows(month_path, rows)
                    nonzero_days += 1
                    month_new_rows += len(rows)

                append_progress(
                    progress_path,
                    {
                        "api_day": day.isoformat(),
                        "symbol": symbol,
                        "status": "done",
                        "api_found": found,
                        "api_returned": returned,
                        "api_limit_per_day": args.news_limit_per_day,
                        "saved_rows_this_day": len(rows),
                        "month_file": str(month_path),
                        "fetched_at_utc": fetch_ts,
                    },
                )
                completed_days.add(day.isoformat())
            except Exception as exc:  # noqa: BLE001
                append_progress(
                    progress_path,
                    {
                        "api_day": day.isoformat(),
                        "symbol": symbol,
                        "status": "error",
                        "api_found": "",
                        "api_returned": "",
                        "api_limit_per_day": args.news_limit_per_day,
                        "saved_rows_this_day": 0,
                        "month_file": str(month_path),
                        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                        "error": str(exc),
                    },
                )
                if not args.continue_on_news_error:
                    raise
                print(f"  [ERR] {day}: {exc}", flush=True)

            random_sleep(args.news_sleep_min, args.news_sleep_max)

        if month_path.exists():
            rows_total = len(pd.read_csv(month_path))
            print(f"[SAVED] {month_path} rows={rows_total} new_rows={month_new_rows} nonzero_days_this_run={nonzero_days}")
        else:
            print(f"[NO DATA] {m_start:%Y-%m} -> not saving CSV (0 rows)")

    return raw_dir


def build_news_sentiment(args: argparse.Namespace) -> Path:
    _, combined_pipeline = script_paths(args.root)
    slug = symbol_lower(args.symbol)
    raw_dir = args.data_dir / "raw" / f"news_headlines_{slug}"
    master = args.data_dir / "interim" / f"{slug}_news_headlines_master.csv"
    clean = args.data_dir / "processed" / f"{slug}_news_headlines_clean.csv"
    sentiment = args.data_dir / "processed" / f"{slug}_news_daily_sentiment.csv"

    run_cmd([sys.executable, str(combined_pipeline), "news-combine", "--folder", str(raw_dir), "--output", str(master)], args.dry_run)
    run_cmd([sys.executable, str(combined_pipeline), "news-clean", "--input", str(master), "--output", str(clean)], args.dry_run)
    run_cmd([sys.executable, str(combined_pipeline), "news-sentiment", "--input", str(clean), "--output", str(sentiment)], args.dry_run)
    return sentiment


# -----------------------------------------------------------------------------
# Model dataset and reporting
# -----------------------------------------------------------------------------
def build_model(args: argparse.Namespace) -> Path:
    _, combined_pipeline = script_paths(args.root)
    symbol = args.symbol.upper()
    slug = symbol_lower(symbol)
    output = args.data_dir / "model_feed" / f"{slug}_model_dataset.csv"
    target = processed_stock_path(args.data_dir, symbol)
    trends = args.data_dir / "processed" / f"{slug}_trends_processed.csv"
    sentiment = args.data_dir / "processed" / f"{slug}_news_daily_sentiment.csv"

    macro_paths = {m.upper(): processed_stock_path(args.data_dir, m.upper()) for m in args.macro_symbols}
    run_cmd(
        [
            sys.executable,
            str(combined_pipeline),
            "build-model",
            "--nvda",
            str(target),
            "--spy",
            str(macro_paths["SPY"]),
            "--soxx",
            str(macro_paths["SOXX"]),
            "--ief",
            str(macro_paths["IEF"]),
            "--sentiment",
            str(sentiment),
            "--trends",
            str(trends),
            "--output",
            str(output),
        ],
        args.dry_run,
    )
    return output


def validate_and_audit(args: argparse.Namespace) -> None:
    _, combined_pipeline = script_paths(args.root)
    slug = symbol_lower(args.symbol)
    model = args.data_dir / "model_feed" / f"{slug}_model_dataset.csv"
    audit = args.data_dir / "model_feed" / f"{slug}_model_dataset_audit.xlsx"
    run_cmd([sys.executable, str(combined_pipeline), "validate", "--input", str(model)], args.dry_run)
    run_cmd([sys.executable, str(combined_pipeline), "audit", "--input", str(model), "--output", str(audit)], args.dry_run)


def run_all(args: argparse.Namespace) -> None:
    fetch_eod(args, args.symbol, macro=False)
    clean_eod(args, args.symbol, macro=False)

    for macro_symbol in args.macro_symbols:
        macro_path = processed_stock_path(args.data_dir, macro_symbol.upper())
        raw_path = raw_stock_path(args.data_dir, macro_symbol.upper(), args.start, args.end, macro=True)
        if args.refresh_macro or not macro_path.exists():
            fetch_eod(args, macro_symbol.upper(), macro=True)
            clean_eod(args, macro_symbol.upper(), macro=True)
        else:
            print(f"[SKIP] Macro processed exists for {macro_symbol}: {macro_path}")
            if not raw_path.exists():
                print(f"       Raw macro path for this exact period not found, but processed file exists: {macro_path}")

    fetch_google_trends(args)
    clean_trends(args)
    fetch_news(args)
    build_news_sentiment(args)
    build_model(args)
    validate_and_audit(args)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    data_dir = Path(os.getenv("THESIS_DATA_DIR", root / "data")).resolve()

    ap = argparse.ArgumentParser(description="Generic stock pipeline for thesis robustness datasets.")
    ap.add_argument(
        "command",
        choices=[
            "all",
            "fetch-stock",
            "clean-stock",
            "fetch-trends",
            "clean-trends",
            "fetch-news",
            "build-sentiment",
            "build-model",
            "validate-audit",
        ],
    )
    ap.add_argument("--symbol", required=True, help="Target stock ticker, e.g. NVDA or AMD.")
    ap.add_argument("--keyword", default=None, help="Google Trends keyword, default: '<SYMBOL> stock'.")
    ap.add_argument("--start", default="2019-03-01", help="Market/Trends start date YYYY-MM-DD.")
    ap.add_argument("--end", default="2026-03-01", help="Market/Trends end date YYYY-MM-DD.")
    ap.add_argument("--news-start", default=None, help="News start date; defaults to --start.")
    ap.add_argument("--news-end", default=None, help="News end date; defaults to --end.")
    ap.add_argument("--root", type=Path, default=root)
    ap.add_argument("--data-dir", type=Path, default=data_dir)
    ap.add_argument("--macro-symbols", nargs="+", default=["SPY", "SOXX", "IEF"])
    ap.add_argument("--refresh-macro", action="store_true", help="Fetch and clean macro symbols for this exact date range.")
    ap.add_argument("--force", action="store_true", help="Refetch/rebuild even when checkpoint/output exists.")
    ap.add_argument("--dry-run", action="store_true")

    ap.add_argument("--news-limit-per-day", type=int, default=10)
    ap.add_argument("--news-sleep-min", type=float, default=0.4)
    ap.add_argument("--news-sleep-max", type=float, default=1.4)
    ap.add_argument("--news-retries", type=int, default=5)
    ap.add_argument("--continue-on-news-error", action="store_true")

    ap.add_argument("--trends-chunk-days", type=int, default=90)
    ap.add_argument("--trends-sleep-min", type=float, default=8.0)
    ap.add_argument("--trends-sleep-max", type=float, default=20.0)
    ap.add_argument("--trends-retries", type=int, default=4)
    ap.add_argument("--trends-hl", default="en-US")
    ap.add_argument("--trends-tz", type=int, default=360)
    ap.add_argument("--trends-geo", default="")
    ap.add_argument("--trends-gprop", default="")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.root = args.root.resolve()
    args.data_dir = args.data_dir.resolve()

    if args.command == "all":
        run_all(args)
    elif args.command == "fetch-stock":
        fetch_eod(args, args.symbol, macro=False)
    elif args.command == "clean-stock":
        clean_eod(args, args.symbol, macro=False)
    elif args.command == "fetch-trends":
        fetch_google_trends(args)
    elif args.command == "clean-trends":
        clean_trends(args)
    elif args.command == "fetch-news":
        fetch_news(args)
    elif args.command == "build-sentiment":
        build_news_sentiment(args)
    elif args.command == "build-model":
        build_model(args)
    elif args.command == "validate-audit":
        validate_and_audit(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
