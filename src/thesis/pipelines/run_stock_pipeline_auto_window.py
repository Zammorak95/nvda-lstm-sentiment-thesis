#!/usr/bin/env python3
"""
Auto-window wrapper around run_stock_pipeline.py.

The limiting source for a sentiment-augmented stock dataset is often StockData
news coverage. This wrapper discovers the earliest date on which symbol-filtered
news is available, then uses that as the news/model window start. To avoid a
large number of API calls, the scan is hierarchical: year probes first, then
month probes inside the first positive year, then daily probes inside the first
positive month.

The collection end date defaults to yesterday in Europe/Amsterdam. Raw and
intermediate data are still checkpointed by run_stock_pipeline.py. The news
availability scan itself is also saved so it can be resumed and audited.

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


def month_start_end(d: date) -> tuple[date, date]:
    start = d.replace(day=1)
    if d.month == 12:
        next_month = d.replace(year=d.year + 1, month=1, day=1)
    else:
        next_month = d.replace(month=d.month + 1, day=1)
    return start, next_month - timedelta(days=1)


def month_iter(start: date, end: date):
    cur = start.replace(day=1)
    while cur <= end:
        m_start, m_end = month_start_end(cur)
        yield max(m_start, start), min(m_end, end)
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1, day=1)
        else:
            cur = cur.replace(month=cur.month + 1, day=1)


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


def get_header(headers: requests.structures.CaseInsensitiveDict, name: str) -> str:
    return headers.get(name, "") or headers.get(name.lower(), "") or headers.get(name.upper(), "") or ""


def response_limit_headers(response: requests.Response) -> dict[str, str]:
    return {
        "x_ratelimit_limit": get_header(response.headers, "X-RateLimit-Limit"),
        "x_ratelimit_remaining": get_header(response.headers, "X-RateLimit-Remaining"),
        "x_usagelimit_limit": get_header(response.headers, "X-UsageLimit-Limit"),
        "x_usagelimit_remaining": get_header(response.headers, "X-UsageLimit-Remaining"),
    }


def scan_news_query(
    session: requests.Session,
    token: str,
    symbol: str,
    limit: int,
    *,
    published_on: date | None = None,
    published_after: date | None = None,
    published_before: date | None = None,
) -> tuple[int, int, dict[str, str]]:
    params = {
        "api_token": token,
        "symbols": symbol.upper(),
        "filter_entities": "true",
        "language": "en",
        "limit": str(limit),
        "page": "1",
    }

    if published_on is not None:
        params["published_on"] = published_on.isoformat()
    else:
        if published_after is None or published_before is None:
            raise ValueError("Range probes require published_after and published_before.")
        params["published_after"] = published_after.isoformat()
        # Add one day outside this function when an inclusive end is desired.
        params["published_before"] = published_before.isoformat()

    response = session.get(base.STOCKDATA_NEWS_URL, params=params, timeout=30)
    limit_headers = response_limit_headers(response)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(f"StockData error: {payload['error']}")
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    articles = payload.get("data", []) if isinstance(payload, dict) else []
    found = int(meta.get("found", 0) or 0)
    returned = len(articles)
    return found, returned, limit_headers


SCAN_COLUMNS = [
    "scan_level",
    "range_start",
    "range_end",
    "symbol",
    "found",
    "returned",
    "limit",
    "status",
    "error",
    "x_ratelimit_limit",
    "x_ratelimit_remaining",
    "x_usagelimit_limit",
    "x_usagelimit_remaining",
    "scanned_at_utc",
]


def read_scan_progress(progress: Path) -> pd.DataFrame:
    if not progress.exists():
        return pd.DataFrame(columns=SCAN_COLUMNS)
    df = pd.read_csv(progress)
    for col in SCAN_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[SCAN_COLUMNS]


def write_scan_progress(progress: Path, df: pd.DataFrame) -> None:
    progress.parent.mkdir(parents=True, exist_ok=True)
    df = df[SCAN_COLUMNS].drop_duplicates(subset=["scan_level", "range_start", "range_end", "symbol"], keep="last")
    df = df.sort_values(["scan_level", "range_start", "range_end"])
    df.to_csv(progress, index=False)


def cached_or_probe(
    *,
    args: argparse.Namespace,
    session: requests.Session,
    token: str,
    progress: Path,
    df_progress: pd.DataFrame,
    level: str,
    start: date,
    end: date,
) -> tuple[int, int, pd.DataFrame]:
    symbol = args.symbol.upper()
    start_s = start.isoformat()
    end_s = end.isoformat()

    mask = (
        df_progress["scan_level"].astype(str).eq(level)
        & df_progress["range_start"].astype(str).eq(start_s)
        & df_progress["range_end"].astype(str).eq(end_s)
        & df_progress["symbol"].astype(str).eq(symbol)
        & df_progress["status"].astype(str).eq("done")
    )
    if mask.any() and not args.force_scan:
        row = df_progress.loc[mask].iloc[-1]
        found = int(float(row.get("found", 0) or 0))
        returned = int(float(row.get("returned", 0) or 0))
        print(f"  [CACHE {level.upper()}] {start_s} -> {end_s} found={found} returned={returned}")
        return found, returned, df_progress

    for attempt in range(1, args.scan_retries + 1):
        try:
            if level == "day":
                found, returned, headers = scan_news_query(
                    session,
                    token,
                    symbol,
                    args.news_limit_per_day,
                    published_on=start,
                )
            else:
                # StockData supports published_after/published_before for range queries.
                # Use end + 1 day so the range is effectively inclusive of the end date.
                found, returned, headers = scan_news_query(
                    session,
                    token,
                    symbol,
                    args.news_limit_per_day,
                    published_after=start,
                    published_before=end + timedelta(days=1),
                )

            row = {
                "scan_level": level,
                "range_start": start_s,
                "range_end": end_s,
                "symbol": symbol,
                "found": found,
                "returned": returned,
                "limit": args.news_limit_per_day,
                "status": "done",
                "error": "",
                **headers,
                "scanned_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            df_progress = pd.concat([df_progress, pd.DataFrame([row])], ignore_index=True)
            write_scan_progress(progress, df_progress)
            print(f"  [SCAN {level.upper()}] {start_s} -> {end_s} found={found} returned={returned}")
            return found, returned, df_progress
        except Exception as exc:  # noqa: BLE001
            if attempt >= args.scan_retries:
                row = {
                    "scan_level": level,
                    "range_start": start_s,
                    "range_end": end_s,
                    "symbol": symbol,
                    "found": "",
                    "returned": "",
                    "limit": args.news_limit_per_day,
                    "status": "error",
                    "error": str(exc),
                    "x_ratelimit_limit": "",
                    "x_ratelimit_remaining": "",
                    "x_usagelimit_limit": "",
                    "x_usagelimit_remaining": "",
                    "scanned_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                df_progress = pd.concat([df_progress, pd.DataFrame([row])], ignore_index=True)
                write_scan_progress(progress, df_progress)
                if not args.continue_on_scan_error:
                    raise
                print(f"  [SCAN ERR] {level} {start_s}->{end_s}: {exc}")
                return 0, 0, df_progress

            wait = random.uniform(args.scan_sleep_min, args.scan_sleep_max) * attempt
            print(f"  [WARN] {level} {start_s}->{end_s} attempt {attempt} failed: {exc}; sleep {wait:.1f}s")
            time.sleep(wait)

    return 0, 0, df_progress


def discover_first_news_date(args: argparse.Namespace) -> str:
    """Find earliest day with StockData symbol-filtered news using year->month->day probes."""
    token = get_stockdata_token()
    symbol = args.symbol.upper()
    start = parse_date(args.scan_start)
    end = parse_date(args.end)
    if end < start:
        raise ValueError(f"--end ({end}) must be >= --scan-start ({start})")

    progress = scan_progress_path(args.data_dir, symbol)
    df_progress = read_scan_progress(progress)

    # If an exact positive day was already found, reuse it immediately.
    if not df_progress.empty and not args.force_scan:
        day_rows = df_progress[
            df_progress["scan_level"].astype(str).eq("day")
            & df_progress["symbol"].astype(str).eq(symbol)
            & df_progress["status"].astype(str).eq("done")
        ].copy()
        if not day_rows.empty:
            day_rows["found_num"] = pd.to_numeric(day_rows["found"], errors="coerce").fillna(0)
            hits = day_rows[day_rows["found_num"] > 0]
            if not hits.empty:
                first = str(hits.sort_values("range_start").iloc[0]["range_start"])
                print(f"[SCAN] Existing first exact news date for {symbol}: {first}")
                return first

    session = requests.Session()
    session.headers.update({"User-Agent": "thesis-news-availability-scan/1.0"})

    print(f"[SCAN] Hierarchical news availability scan for {symbol}: {start} -> {end}")
    print("[SCAN] Strategy: year probes -> month probes -> day probes, to minimize API calls.")

    first_year: tuple[date, date] | None = None
    for year in range(start.year, end.year + 1):
        y_start = max(date(year, 1, 1), start)
        y_end = min(date(year, 12, 31), end)
        found, returned, df_progress = cached_or_probe(
            args=args,
            session=session,
            token=token,
            progress=progress,
            df_progress=df_progress,
            level="year",
            start=y_start,
            end=y_end,
        )
        if found > 0 or returned > 0:
            first_year = (y_start, y_end)
            break
        random_sleep(args.scan_sleep_min, args.scan_sleep_max)

    if first_year is None:
        raise RuntimeError(f"No StockData news found for {symbol} between {start} and {end}.")

    first_month: tuple[date, date] | None = None
    for m_start, m_end in month_iter(first_year[0], first_year[1]):
        found, returned, df_progress = cached_or_probe(
            args=args,
            session=session,
            token=token,
            progress=progress,
            df_progress=df_progress,
            level="month",
            start=m_start,
            end=m_end,
        )
        if found > 0 or returned > 0:
            first_month = (m_start, m_end)
            break
        random_sleep(args.scan_sleep_min, args.scan_sleep_max)

    if first_month is None:
        raise RuntimeError(f"Year probe found news for {symbol}, but no positive month was identified.")

    for day in date_range(first_month[0], first_month[1]):
        found, returned, df_progress = cached_or_probe(
            args=args,
            session=session,
            token=token,
            progress=progress,
            df_progress=df_progress,
            level="day",
            start=day,
            end=day,
        )
        if found > 0 or returned > 0:
            print(f"[SCAN] First exact news date for {symbol}: {day}")
            print(f"[SCAN] Progress saved to: {progress}")
            return day.isoformat()
        random_sleep(args.scan_sleep_min, args.scan_sleep_max)

    raise RuntimeError(f"Month probe found news for {symbol}, but no exact positive day was identified.")


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

    ap.add_argument("--scan-sleep-min", type=float, default=0.8)
    ap.add_argument("--scan-sleep-max", type=float, default=2.5)
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
