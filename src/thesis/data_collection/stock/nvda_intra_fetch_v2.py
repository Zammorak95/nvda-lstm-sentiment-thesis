#!/usr/bin/env python3
"""Fetch NVDA intraday *minute* data from StockData.org in 7-day chunks.

Notes:
- StockData docs: minute interval has a max range of 7 days per request.
- StockData Free plan lists ~1 month of intraday history, so defaults use last 30 days.

Auth env vars (either works):
- STOCKDATA_API_KEY
- STOCKDATA_API_TOKEN
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd
import requests

API_URL = "https://api.stockdata.org/v1/data/intraday"


def get_api_token() -> str:
    for name in ("STOCKDATA_API_KEY", "STOCKDATA_API_TOKEN"):
        val = os.getenv(name)
        if val:
            return val
    raise RuntimeError("Missing API token. Export STOCKDATA_API_TOKEN or STOCKDATA_API_KEY.")


def to_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def fetch_chunk(symbol: str, start: str, end: str, token: str) -> tuple[pd.DataFrame, dict]:
    params = {
        "api_token": token,
        "symbols": symbol,
        "date_from": start,
        "date_to": end,
        "interval": "minute",
        "sort": "asc",
    }
    r = requests.get(API_URL, params=params, timeout=90)
    r.raise_for_status()
    payload = r.json()

    if isinstance(payload, dict) and (payload.get("error") or payload.get("errors")):
        raise RuntimeError(f"API error: {payload.get('error') or payload.get('errors')}")

    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    data = payload.get("data", []) if isinstance(payload, dict) else []
    df = pd.DataFrame(data)

    # Intraday schema may nest OHLCV under 'data'
    if not df.empty and "data" in df.columns and isinstance(df.loc[df.index[0], "data"], dict):
        nested = pd.json_normalize(df["data"])
        df = pd.concat([df.drop(columns=["data"]), nested], axis=1)

    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None)
        df = df.sort_values("date").reset_index(drop=True)

    return df, meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="NVDA")
    p.add_argument("--start", default="2026-01-31")
    p.add_argument("--end", default=str(date.today()))
    p.add_argument("--outdir", default="./data")
    p.add_argument("--chunk_days", type=int, default=7)
    p.add_argument("--sleep", type=float, default=1.0)
    p.add_argument("--csv", action="store_true")
    args = p.parse_args()

    token = get_api_token()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    start_d = to_date(args.start)
    end_d = to_date(args.end)

    all_dfs: List[pd.DataFrame] = []
    cur = start_d
    while cur <= end_d:
        chunk_end = min(end_d, cur + timedelta(days=args.chunk_days - 1))
        s = cur.strftime("%Y-%m-%d")
        e = chunk_end.strftime("%Y-%m-%d")

        print(f"Fetching {args.symbol} minute: {s}..{e}")
        df, meta = fetch_chunk(args.symbol, s, e, token)
        print(f"  rows={len(df):,} meta={meta}")
        if not df.empty:
            all_dfs.append(df)

        cur = chunk_end + timedelta(days=1)
        if cur <= end_d and args.sleep > 0:
            time.sleep(args.sleep)

    if not all_dfs:
        print("No intraday data returned. Try a more recent range (Free plan ~1 month).")
        return

    out = pd.concat(all_dfs, ignore_index=True)
    if "date" in out.columns:
        out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    pq = outdir / f"{args.symbol}_intraday_minute_{args.start}_to_{args.end}.parquet"
    out.to_parquet(pq, index=False)
    print(f"Wrote {len(out):,} rows -> {pq}")

    if args.csv:
        csvp = outdir / f"{args.symbol}_intraday_minute_{args.start}_to_{args.end}.csv"
        out.to_csv(csvp, index=False)
        print(f"Wrote {len(out):,} rows -> {csvp}")


if __name__ == "__main__":
    main()
