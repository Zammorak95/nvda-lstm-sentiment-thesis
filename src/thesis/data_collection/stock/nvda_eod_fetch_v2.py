#!/usr/bin/env python3
"""Fetch NVDA EOD data from StockData.org and save to Parquet (optionally CSV).

Defaults are set to work on the *Free* StockData plan (which lists ~1 month of EOD data).
If you request older dates, the API may return an empty `data` array.

Auth env vars (either works):
- STOCKDATA_API_KEY
- STOCKDATA_API_TOKEN
"""

from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://api.stockdata.org/v1/data/eod"


def get_api_token() -> str:
    for name in ("STOCKDATA_API_KEY", "STOCKDATA_API_TOKEN"):
        val = os.getenv(name)
        if val:
            return val
    raise RuntimeError("Missing API token. Export STOCKDATA_API_TOKEN or STOCKDATA_API_KEY.")


def fetch_eod(symbol: str, start: str, end: str, token: str) -> tuple[pd.DataFrame, dict]:
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

    # Handle explicit error payloads
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
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="NVDA")
    p.add_argument("--start", default="2026-01-31")
    p.add_argument("--end", default=str(date.today()))
    p.add_argument("--outdir", default="./data")
    p.add_argument("--csv", action="store_true")
    args = p.parse_args()

    token = get_api_token()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df, meta = fetch_eod(args.symbol, args.start, args.end, token)

    if df.empty:
        print(f"No EOD data returned. meta={meta}")
        print("Try a more recent range (Free plan ~1 month), e.g. --start $(date -I -d '30 days ago')")
        return

    pq = outdir / f"{args.symbol}_eod_{args.start}_to_{args.end}.parquet"
    df.to_parquet(pq, index=False)
    print(f"Wrote {len(df):,} rows -> {pq}")

    if args.csv:
        csvp = outdir / f"{args.symbol}_eod_{args.start}_to_{args.end}.csv"
        df.to_csv(csvp, index=False)
        print(f"Wrote {len(df):,} rows -> {csvp}")


if __name__ == "__main__":
    main()

