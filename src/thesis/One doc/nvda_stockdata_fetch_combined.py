#!/usr/bin/env python3
"""
nvda_stockdata_fetch_combined.py

Fetch NVIDIA (NVDA by default) market data from StockData.org and save to Parquet
(optionally CSV).

This combines the uploaded EOD and intraday fetchers into one CLI:
- --mode eod: fetch End-of-Day data, using chunking by default.
- --mode intraday: fetch intraday data, minute interval by default, using chunks.

Auth env vars (either works):
- STOCKDATA_API_TOKEN
- STOCKDATA_API_KEY

Examples:
  # EOD, recent range, also write CSV
  python nvda_stockdata_fetch_combined.py --mode eod --start 2026-02-01 --end 2026-03-01 --csv

  # EOD, long range, chunked using API max_period_days when available
  python nvda_stockdata_fetch_combined.py --mode eod --start 2016-03-01 --end 2026-03-01 --csv

  # Intraday minute data, chunked into 7-day windows
  python nvda_stockdata_fetch_combined.py --mode intraday --start 2026-02-01 --end 2026-03-01 --csv

  # Different symbol
  python nvda_stockdata_fetch_combined.py --mode eod --symbol SPY --start 2026-02-01 --csv
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

EOD_API_URL = "https://api.stockdata.org/v1/data/eod"
INTRADAY_API_URL = "https://api.stockdata.org/v1/data/intraday"


def get_api_token() -> str:
    """Return the StockData API token from the environment."""
    for name in ("STOCKDATA_API_TOKEN", "STOCKDATA_API_KEY"):
        val = os.getenv(name)
        if val:
            return val
    raise RuntimeError(
        "Missing API token. Export STOCKDATA_API_TOKEN or STOCKDATA_API_KEY "
        "(for example: source .env)."
    )


def to_date(s: str) -> date:
    """Parse YYYY-MM-DD into a date."""
    return datetime.strptime(s, "%Y-%m-%d").date()


def request_stockdata(url: str, params: dict[str, Any], timeout: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Call StockData, validate the response, and return (dataframe, meta)."""
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    payload = r.json()

    if isinstance(payload, dict) and (payload.get("error") or payload.get("errors")):
        raise RuntimeError(f"API error: {payload.get('error') or payload.get('errors')}")

    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    data = payload.get("data", []) if isinstance(payload, dict) else []
    df = pd.DataFrame(data)
    return df, meta


def normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Parse, sort, and normalize the StockData date column when present."""
    if not df.empty and "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None)
        df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_eod_window(symbol: str, start: str, end: str, token: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    params = {
        "api_token": token,
        "symbols": symbol,
        "date_from": start,
        "date_to": end,
        "sort": "asc",
    }
    df, meta = request_stockdata(EOD_API_URL, params, timeout=60)
    return normalize_date_column(df), meta


def fetch_intraday_window(
    symbol: str,
    start: str,
    end: str,
    token: str,
    interval: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    params = {
        "api_token": token,
        "symbols": symbol,
        "date_from": start,
        "date_to": end,
        "interval": interval,
        "sort": "asc",
    }
    df, meta = request_stockdata(INTRADAY_API_URL, params, timeout=90)

    # Intraday schema may nest OHLCV under a column named 'data'.
    if not df.empty and "data" in df.columns and isinstance(df.loc[df.index[0], "data"], dict):
        nested = pd.json_normalize(df["data"])
        df = pd.concat([df.drop(columns=["data"]), nested], axis=1)

    return normalize_date_column(df), meta


def write_outputs(
    df: pd.DataFrame,
    outdir: Path,
    symbol: str,
    mode: str,
    start: str,
    end: str,
    csv: bool,
    interval: str | None = None,
) -> None:
    """Write Parquet, and CSV when requested."""
    descriptor = "eod" if mode == "eod" else f"intraday_{interval}"
    pq = outdir / f"{symbol}_{descriptor}_{start}_to_{end}.parquet"
    df.to_parquet(pq, index=False)
    print(f"Wrote {len(df):,} rows -> {pq}")

    if csv:
        csvp = outdir / f"{symbol}_{descriptor}_{start}_to_{end}.csv"
        df.to_csv(csvp, index=False)
        print(f"Wrote {len(df):,} rows -> {csvp}")


def combine_windows(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate, sort, and deduplicate window results."""
    out = pd.concat(dfs, ignore_index=True)
    if "date" in out.columns:
        out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return out


def run_eod(args: argparse.Namespace, token: str, outdir: Path) -> None:
    start_d = to_date(args.start)
    end_d = to_date(args.end)
    if end_d < start_d:
        raise ValueError("--end must be >= --start")

    if args.max_days and args.max_days > 0:
        effective_max_days = args.max_days
        print(f"Using overridden max_days={effective_max_days}")
    else:
        probe_end = min(end_d, start_d + timedelta(days=1))
        _, meta = fetch_eod_window(args.symbol, start_d.strftime("%Y-%m-%d"), probe_end.strftime("%Y-%m-%d"), token)
        effective_max_days = int(meta.get("max_period_days") or 180)
        print(f"Using API max_period_days={effective_max_days} (fallback 180 if missing). meta={meta}")

    all_dfs: list[pd.DataFrame] = []
    cur = start_d
    empty_windows = 0

    while cur <= end_d:
        win_end = min(end_d, cur + timedelta(days=effective_max_days - 1))
        s = cur.strftime("%Y-%m-%d")
        e = win_end.strftime("%Y-%m-%d")

        print(f"Fetching {args.symbol} EOD: {s} .. {e}")
        df, meta = fetch_eod_window(args.symbol, s, e, token)
        print(f"  rows={len(df):,} meta={meta}")

        if df.empty:
            empty_windows += 1
            if not args.continue_on_empty:
                print(
                    "Stopping because this EOD window returned 0 rows. "
                    "This usually means your plan does not include data for this period, "
                    "or the symbol/date range is invalid."
                )
                break
        else:
            all_dfs.append(df)

        cur = win_end + timedelta(days=1)
        if cur <= end_d and args.sleep > 0:
            time.sleep(args.sleep)

    if not all_dfs:
        print(f"No EOD data downloaded for {args.symbol}. Empty windows encountered: {empty_windows}.")
        return

    out = combine_windows(all_dfs)
    write_outputs(out, outdir, args.symbol, "eod", args.start, args.end, args.csv)


def run_intraday(args: argparse.Namespace, token: str, outdir: Path) -> None:
    start_d = to_date(args.start)
    end_d = to_date(args.end)
    if end_d < start_d:
        raise ValueError("--end must be >= --start")
    if args.chunk_days <= 0:
        raise ValueError("--chunk_days must be > 0")

    all_dfs: list[pd.DataFrame] = []
    cur = start_d

    while cur <= end_d:
        chunk_end = min(end_d, cur + timedelta(days=args.chunk_days - 1))
        s = cur.strftime("%Y-%m-%d")
        e = chunk_end.strftime("%Y-%m-%d")

        print(f"Fetching {args.symbol} {args.interval}: {s} .. {e}")
        df, meta = fetch_intraday_window(args.symbol, s, e, token, args.interval)
        print(f"  rows={len(df):,} meta={meta}")
        if not df.empty:
            all_dfs.append(df)
        elif not args.continue_on_empty:
            print(
                "Stopping because this intraday window returned 0 rows. "
                "Try a more recent range if your plan only includes recent intraday history."
            )
            break

        cur = chunk_end + timedelta(days=1)
        if cur <= end_d and args.sleep > 0:
            time.sleep(args.sleep)

    if not all_dfs:
        print(f"No intraday data downloaded for {args.symbol}.")
        return

    out = combine_windows(all_dfs)
    write_outputs(out, outdir, args.symbol, "intraday", args.start, args.end, args.csv, interval=args.interval)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fetch StockData.org EOD or intraday data in chunks (default symbol: NVDA)."
    )
    p.add_argument("--mode", choices=("eod", "intraday"), default="eod", help="Data type to fetch (default: eod)")
    p.add_argument("--symbol", default="NVDA", help="Ticker symbol (default: NVDA)")
    p.add_argument("--start", default="2026-01-31", help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=str(date.today()), help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--outdir", default="./data", help="Output directory (default: ./data)")
    p.add_argument("--csv", action="store_true", help="Also write CSV (Parquet is always written)")
    p.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between requests (default: 0.5)")
    p.add_argument(
        "--continue_on_empty",
        action="store_true",
        help="Continue fetching subsequent windows even if one window returns 0 rows.",
    )

    eod = p.add_argument_group("EOD options")
    eod.add_argument(
        "--max_days",
        type=int,
        default=0,
        help="EOD max days per request. If 0, use API meta.max_period_days when available (else 180).",
    )

    intraday = p.add_argument_group("Intraday options")
    intraday.add_argument("--interval", default="minute", help="Intraday interval (default: minute)")
    intraday.add_argument("--chunk_days", type=int, default=7, help="Intraday days per request (default: 7)")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    token = get_api_token()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.mode == "eod":
        run_eod(args, token, outdir)
    else:
        run_intraday(args, token, outdir)


if __name__ == "__main__":
    main()
