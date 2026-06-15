#!/usr/bin/env python3
"""
nvda_eod_fetch_chunked.py

Fetch NVIDIA (NVDA) End-of-Day (EOD) data from StockData.org in chunks and save to disk.

Why chunking?
- StockData may enforce a maximum request window (often exposed as `meta.max_period_days`).
- If you request a long range, the API can truncate it and/or return empty data depending on plan limits.

This script:
- Splits the requested date range into windows (default: use API-reported max_period_days if present, else 180).
- Fetches each window sequentially.
- Concatenates and deduplicates by date.
- Writes Parquet and (optionally) CSV.

Auth:
- Uses STOCKDATA_API_TOKEN or STOCKDATA_API_KEY from your environment.
  (So your .env can contain STOCKDATA_API_TOKEN=...)

Examples:
  # last month (works on free plans that only allow recent history)
  python nvda_eod_fetch_chunked.py --start 2026-02-01 --end 2026-03-01 --csv

  # long range (requires plan history that covers the period)
  python nvda_eod_fetch_chunked.py --start 2016-03-01 --end 2026-03-01 --csv

  # keep going even if some chunks return empty (not recommended unless you expect gaps)
  python nvda_eod_fetch_chunked.py --start 2016-03-01 --end 2026-03-01 --continue_on_empty
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd
import requests

API_URL = "https://api.stockdata.org/v1/data/eod"


def get_api_token() -> str:
    for name in ("STOCKDATA_API_TOKEN", "STOCKDATA_API_KEY"):
        val = os.getenv(name)
        if val:
            return val
    raise RuntimeError("Missing API token. Export STOCKDATA_API_TOKEN or STOCKDATA_API_KEY (e.g. via `source .env`).")


def to_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def fetch_window(symbol: str, start: str, end: str, token: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    params = {
        "api_token": token,
        "symbols": symbol,
        "date_from": start,
        "date_to": end,
        "sort": "asc",
    }
    r = requests.get(API_URL, params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()

    if isinstance(payload, dict) and (payload.get("error") or payload.get("errors")):
        raise RuntimeError(f"API error: {payload.get('error') or payload.get('errors')}")

    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    data = payload.get("data", []) if isinstance(payload, dict) else []
    df = pd.DataFrame(data)

    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None)
        df = df.sort_values("date").reset_index(drop=True)

    return df, meta


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch StockData.org EOD data in chunks (default NVDA).")
    p.add_argument("--symbol", default="NVDA", help="Ticker symbol (default: SPY)")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD (required)")
    p.add_argument("--end", default=str(date.today()), help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--outdir", default="./data", help="Output directory (default: ./data)")
    p.add_argument("--csv", action="store_true", help="Also write CSV (Parquet is always written)")
    p.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between requests (default: 0.5)")
    p.add_argument(
        "--max_days",
        type=int,
        default=0,
        help="Override max days per request. If 0, use API meta.max_period_days when available (else 180).",
    )
    p.add_argument(
        "--continue_on_empty",
        action="store_true",
        help="Continue fetching subsequent windows even if a window returns 0 rows (default: stop on first empty).",
    )
    args = p.parse_args()

    token = get_api_token()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    start_d = to_date(args.start)
    end_d = to_date(args.end)
    if end_d < start_d:
        raise ValueError("--end must be >= --start")

    # First probe call (tiny window) to learn API max_period_days (if we didn't override)
    if args.max_days and args.max_days > 0:
        effective_max_days = args.max_days
        print(f"Using overridden max_days={effective_max_days}")
    else:
        probe_end = min(end_d, start_d + timedelta(days=1))
        _, meta = fetch_window(args.symbol, start_d.strftime("%Y-%m-%d"), probe_end.strftime("%Y-%m-%d"), token)
        effective_max_days = int(meta.get("max_period_days") or 180)
        print(f"Using API max_period_days={effective_max_days} (fallback 180 if missing). meta={meta}")

    all_dfs: List[pd.DataFrame] = []
    cur = start_d
    empty_windows = 0

    while cur <= end_d:
        win_end = min(end_d, cur + timedelta(days=effective_max_days - 1))
        s = cur.strftime("%Y-%m-%d")
        e = win_end.strftime("%Y-%m-%d")

        print(f"Fetching {args.symbol} EOD: {s} .. {e}")
        df, meta = fetch_window(args.symbol, s, e, token)
        print(f"  rows={len(df):,} meta={meta}")

        if df.empty:
            empty_windows += 1
            if not args.continue_on_empty:
                print(
                    "Stopping because this window returned 0 rows. "
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

    out = pd.concat(all_dfs, ignore_index=True)
    if "date" in out.columns:
        out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    pq = outdir / f"{args.symbol}_eod_chunked_{args.start}_to_{args.end}.parquet"
    out.to_parquet(pq, index=False)
    print(f"Wrote {len(out):,} rows -> {pq}")

    if args.csv:
        csvp = outdir / f"{args.symbol}_eod_chunked_{args.start}_to_{args.end}.csv"
        out.to_csv(csvp, index=False)
        print(f"Wrote {len(out):,} rows -> {csvp}")


if __name__ == "__main__":
    main()
