#!/usr/bin/env python3
"""
Auto-window wrapper around run_stock_pipeline.py.

The limiting source for a sentiment-augmented stock dataset is often StockData
news coverage. This wrapper first discovers the earliest date on which
symbol-filtered news is available, then uses that as the news/model window start.
The collection end date defaults to yesterday in Europe/Amsterdam.

Raw/intermediate data are still checkpointed by run_stock_pipeline.py.
The news availability scan itself is also saved so it can be resumed.

Examples:
  python -u run_stock_pipeline_auto_window.py all --symbol AMD --keyword "AMD stock" --scan-start 2018-01-01
  python -u run_stock_pipeline_auto_window.py scan-news --symbol AMD --scan-start 2018-01-01
  python -u run_stock_pipeline_auto_window.py all --symbol TSM --keyword "TSM stock" --scan-start 2019-01-01 --end 2026-02-26
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import requests

# Import the generic pipeline from the same directory when this file is executed
# as a script.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import run_stock_pipeline as base  # noqa: E402


AMSTERDAM_TZ = ZoneInfo("Europe/Amsterdam")


def yesterday_amsterdam() -> str:
    return (datetime.now(AMSTERDAM_TZ).date() - timedelta(days=1)).isoformat()


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def date_range(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def get_stockdata_token() -> str:
    for name in ("STOCKDATA_API_TOKEN", "STOCKDATA_API_KEY"):
        token = os.getenv(name)
        if token:
            return token
    raise RuntimeError("Missing StockData token. Run: set -a; source .env; set +a")


def random_sleep(min_seconds: float, max_seconds: float) -> None:
    if max_seconds <= 0:
        return
    lo = max(0.0, min_seconds)
    hi = max(lo, max_seconds)
    time.sleep(random.uniform(lo, hi))


def scan_progress_path(data_dir: Path, symbol: str) -> Path:
    slug = symbol.lower()
    return data_dir / "raw" / f"news_headlines_{slug}" / f"{symbol.upper()}_news_availability_scan.csv"


def scan_one_day(
    session: requests.Session,
    token: str,
    symbol: str,
    day: date,
    limit: int,
) -> tuple[int, int]:
    params = {
        "api_token": token,
        "symbols": symbol.upper(),
        "filter_entities": "true",
        "language": "en",
        "published_on": day.isoformat(),
        "limit": str(limit),
        "page": "1",
    }
    response = session.get(base.STOCKDATA_NEWS_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(f"StockData error: {payload['error']}")
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    articles = payload.get("data", []) if isinstance(payload, dict) else []
    found = int(meta.get("found", 0) or 0)
    returned = len(articles)
    return found, returned


def discover_first_news_date(args: argparse.Namespace) -> str:
    """Find earliest day with StockData symbol-filtered news and checkpoint each day."""
    token = get_stockdata_token()
    symbol = args.symbol.upper()
    start = parse_date(args.scan_start)
    end = parse_date(args.end)
    if end < start:
        raise ValueError(f"--end ({end}) must be >= --scan-start ({start})")

    progress = scan_progress_path(args.data_dir, symbol)
    progress.parent.mkdir(parents=True, exist_ok=True)

    existing = pd.DataFrame()
    if progress.exists() and not args.force_scan:
        existing = pd.read_csv(progress)
        if not existing.empty and "found" in existing.columns:
            hits = existing[pd.to_numeric(existing["found"], errors="coerce").fillna(0) > 0]
            if not hits.empty:
                first = str(hits.sort_values("date").iloc[0]["date"])
                print(f"[SCAN] Existing first news date for {symbol}: {first}")
                return first

    completed = set(existing["date"].astype(str)) if not existing.empty and "date" in existing.columns and not args.force_scan else set()
    session = requests.Session()
    session.headers.update({"User-Agent": "thesis-news-availability-scan/1.0"})

    field_order = [
        "date",
        "symbol",
        "found",
        "returned",
        "limit",
        "status",
        "error",
        "scanned_at_utc",
    ]

    def append_row(row: dict) -> None:
        row = {key: row.get(key, "") for key in field_order}
        df_row = pd.DataFrame([row])
        if progress.exists() and not args.force_scan:
            df_old = pd.read_csv(progress)
            df = pd.concat([df_old, df_row], ignore_index=True)
        else:
            df = df_row
        df = df.drop_duplicates(subset=["date", "symbol"], keep="last").sort_values("date")
        df.to_csv(progress, index=False)

    print(f"[SCAN] Searching first StockData news date for {symbol}: {start} -> {end}")
    for day in date_range(start, end):
        if day.isoformat() in completed:
            continue

        try:
            found, returned = 0, 0
            last_exc: Exception | None = None
            for attempt in range(1, args.scan_retries + 1):
                try:
                    found, returned = scan_one_day(session, token, symbol, day, args.news_limit_per_day)
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if attempt >= args.scan_retries:
                        raise
                    wait = random.uniform(args.scan_sleep_min, args.scan_sleep_max) * attempt
                    print(f"  [WARN] {day} attempt {attempt} failed: {exc}; sleep {wait:.1f}s")
                    time.sleep(wait)

            append_row(
                {
                    "date": day.isoformat(),
                    "symbol": symbol,
                    "found": found,
                    "returned": returned,
                    "limit": args.news_limit_per_day,
                    "status": "done",
                    "error": "" if last_exc is None else str(last_exc),
                    "scanned_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            print(f"  [SCAN DAY] {day} found={found} returned={returned}")
            if found > 0 or returned > 0:
                print(f"[SCAN] First news date for {symbol}: {day}")
                print(f"[SCAN] Progress saved to: {progress}")
                return day.isoformat()
        except Exception as exc:  # noqa: BLE001
            append_row(
                {
                    "date": day.isoformat(),
                    "symbol": symbol,
                    "found": "",
                    "returned": "",
                    "limit": args.news_limit_per_day,
                    "status": "error",
                    "error": str(exc),
                    "scanned_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            if not args.continue_on_scan_error:
                raise
            print(f"  [SCAN ERR] {day}: {exc}")

        random_sleep(args.scan_sleep_min, args.scan_sleep_max)

    raise RuntimeError(f"No StockData news found for {symbol} between {start} and {end}.")


def build_base_args(args: argparse.Namespace, news_start: str) -> SimpleNamespace:
    """Create the namespace expected by run_stock_pipeline.py."""
    news_start_date = parse_date(news_start)
    fetch_start_date = news_start_date - timedelta(days=args.market_buffer_days)
    market_start = args.market_start or fetch_start_date.isoformat()

    return SimpleNamespace(
        command=args.command,
        symbol=args.symbol.upper(),
        keyword=args.keyword,
        start=market_start,
        end=args.end,
        news_start=news_start,
        news_end=args.end,
        root=args.root,
        data_dir=args.data_dir,
        macro_symbols=args.macro_symbols,
        refresh_macro=args.refresh_macro,
        force=args.force,
        dry_run=args.dry_run,
        news_limit_per_day=args.news_limit_per_day,
        news_sleep_min=args.news_sleep_min,
        news_sleep_max=args.news_sleep_max,
        news_retries=args.news_retries,
        continue_on_news_error=args.continue_on_news_error,
        trends_chunk_days=args.trends_chunk_days,
        trends_sleep_min=args.trends_sleep_min,
        trends_sleep_max=args.trends_sleep_max,
        trends_retries=args.trends_retries,
        trends_hl=args.trends_hl,
        trends_tz=args.trends_tz,
        trends_geo=args.trends_geo,
        trends_gprop=args.trends_gprop,
    )


def build_parser() -> argparse.ArgumentParser:
    root = base.project_root()
    data_dir = Path(os.getenv("THESIS_DATA_DIR", root / "data")).resolve()

    ap = argparse.ArgumentParser(description="Auto-window wrapper for generic stock thesis pipeline.")
    ap.add_argument(
        "command",
        choices=["all", "scan-news", "fetch-stock", "clean-stock", "fetch-trends", "clean-trends", "fetch-news", "build-sentiment", "build-model", "validate-audit"],
    )
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--keyword", default=None, help="Google Trends keyword; default is '<SYMBOL> stock'.")
    ap.add_argument("--scan-start", default="2018-01-01", help="Earliest date to test for StockData news coverage.")
    ap.add_argument("--end", default=yesterday_amsterdam(), help="End date; default is yesterday in Europe/Amsterdam.")
    ap.add_argument("--news-start", default="auto", help="Use 'auto' to scan, or pass YYYY-MM-DD explicitly.")
    ap.add_argument("--market-start", default=None, help="Optional explicit market/Trends fetch start. Otherwise news_start - buffer.")
    ap.add_argument("--market-buffer-days", type=int, default=45, help="Extra stock/Trends history before news_start for rolling features.")
    ap.add_argument("--root", type=Path, default=root)
    ap.add_argument("--data-dir", type=Path, default=data_dir)
    ap.add_argument("--macro-symbols", nargs="+", default=["SPY", "SOXX", "IEF"])
    ap.add_argument("--refresh-macro", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")

    ap.add_argument("--news-limit-per-day", type=int, default=10)
    ap.add_argument("--news-sleep-min", type=float, default=0.4)
    ap.add_argument("--news-sleep-max", type=float, default=1.4)
    ap.add_argument("--news-retries", type=int, default=5)
    ap.add_argument("--continue-on-news-error", action="store_true")

    ap.add_argument("--scan-sleep-min", type=float, default=0.25)
    ap.add_argument("--scan-sleep-max", type=float, default=0.9)
    ap.add_argument("--scan-retries", type=int, default=4)
    ap.add_argument("--force-scan", action="store_true")
    ap.add_argument("--continue-on-scan-error", action="store_true")

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

    if args.news_start == "auto":
        news_start = discover_first_news_date(args)
    else:
        news_start = args.news_start

    print(f"[WINDOW] symbol={args.symbol.upper()} news_start={news_start} end={args.end}")

    if args.command == "scan-news":
        return

    base_args = build_base_args(args, news_start)
    print(f"[WINDOW] market/trends fetch start={base_args.start} news/model start={base_args.news_start} end={base_args.end}")

    if args.command == "all":
        base.run_all(base_args)
    elif args.command == "fetch-stock":
        base.fetch_eod(base_args, base_args.symbol, macro=False)
    elif args.command == "clean-stock":
        base.clean_eod(base_args, base_args.symbol, macro=False)
    elif args.command == "fetch-trends":
        base.fetch_google_trends(base_args)
    elif args.command == "clean-trends":
        base.clean_trends(base_args)
    elif args.command == "fetch-news":
        base.fetch_news(base_args)
    elif args.command == "build-sentiment":
        base.build_news_sentiment(base_args)
    elif args.command == "build-model":
        base.build_model(base_args)
    elif args.command == "validate-audit":
        base.validate_and_audit(base_args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
