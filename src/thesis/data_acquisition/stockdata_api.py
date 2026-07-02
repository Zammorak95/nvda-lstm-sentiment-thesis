#!/usr/bin/env python3
"""Fetch raw market data and news headlines from StockData.org.

This is the canonical raw-data acquisition entry point for the thesis project.
It replaces the older loose script `nvda_stockdata_fetch_combined.py` in the
public reproduction instructions while keeping the same core behaviour for EOD
and intraday market data.

Authentication:
    Set either STOCKDATA_API_TOKEN or STOCKDATA_API_KEY in the environment.

Examples:
    thesis-fetch-stockdata market --mode eod --symbol NVDA --start 2019-03-01 --end 2026-03-01 --csv
    thesis-fetch-stockdata market --mode eod --symbol SPY  --start 2019-03-01 --end 2026-03-01 --csv
    thesis-fetch-stockdata news --symbols NVDA --start 2019-03-01 --end 2026-03-01 --csv
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

EOD_API_URL = "https://api.stockdata.org/v1/data/eod"
INTRADAY_API_URL = "https://api.stockdata.org/v1/data/intraday"
NEWS_API_URL = "https://api.stockdata.org/v1/news/all"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def get_api_token() -> str:
    """Return the StockData API token from the environment."""
    for name in ("STOCKDATA_API_TOKEN", "STOCKDATA_API_KEY"):
        value = os.getenv(name)
        if value:
            return value
    raise RuntimeError(
        "Missing StockData API token. Set STOCKDATA_API_TOKEN or STOCKDATA_API_KEY first."
    )


def to_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def request_json(url: str, params: dict[str, Any], timeout: int) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and (payload.get("error") or payload.get("errors")):
        raise RuntimeError(f"StockData API error: {payload.get('error') or payload.get('errors')}")
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected StockData API response; expected a JSON object.")
    return payload


def payload_to_frame(payload: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta = payload.get("meta", {}) if isinstance(payload.get("meta", {}), dict) else {}
    data = payload.get("data", [])
    if isinstance(data, dict):
        data = [data]
    if data is None:
        data = []
    df = pd.DataFrame(data)
    return df, meta


def normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    if not df.empty and "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None)
        df = df.sort_values("date").reset_index(drop=True)
    elif not df.empty and "published_at" in df.columns:
        df = df.copy()
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        df = df.sort_values("published_at").reset_index(drop=True)
    return df


def combine_windows(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out = pd.concat(frames, ignore_index=True)
    if "date" in out.columns:
        out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    elif "uuid" in out.columns:
        out = out.drop_duplicates(subset=["uuid"], keep="last")
        if "published_at" in out.columns:
            out = out.sort_values("published_at")
    else:
        out = out.drop_duplicates()
    return out.reset_index(drop=True)


def write_frame(
    df: pd.DataFrame,
    outdir: Path,
    filename_stem: str,
    csv: bool,
    parquet: bool,
    meta: dict[str, Any] | None = None,
) -> None:
    ensure_dir(outdir)
    if parquet:
        parquet_path = outdir / f"{filename_stem}.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"Wrote {len(df):,} rows -> {parquet_path}")
    if csv:
        csv_path = outdir / f"{filename_stem}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Wrote {len(df):,} rows -> {csv_path}")
    if meta:
        meta_path = outdir / f"{filename_stem}.meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        print(f"Wrote metadata -> {meta_path}")


def fetch_market_window(
    mode: str,
    symbol: str,
    start: str,
    end: str,
    token: str,
    interval: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    url = EOD_API_URL if mode == "eod" else INTRADAY_API_URL
    params: dict[str, Any] = {
        "api_token": token,
        "symbols": symbol,
        "date_from": start,
        "date_to": end,
        "sort": "asc",
    }
    if mode == "intraday":
        params["interval"] = interval
    payload = request_json(url, params, timeout=90)
    df, meta = payload_to_frame(payload)

    # Some intraday responses nest OHLCV values inside a column named data.
    if not df.empty and "data" in df.columns and isinstance(df.loc[df.index[0], "data"], dict):
        nested = pd.json_normalize(df["data"])
        df = pd.concat([df.drop(columns=["data"]), nested], axis=1)

    return normalize_date_column(df), meta


def run_market(args: argparse.Namespace) -> None:
    token = get_api_token()
    start_d = to_date(args.start)
    end_d = to_date(args.end)
    if end_d < start_d:
        raise ValueError("--end must be >= --start")

    if args.mode == "eod":
        if args.chunk_days > 0:
            chunk_days = args.chunk_days
        else:
            probe_end = min(end_d, start_d + timedelta(days=1))
            _, meta = fetch_market_window(
                "eod",
                args.symbol,
                start_d.strftime("%Y-%m-%d"),
                probe_end.strftime("%Y-%m-%d"),
                token,
                args.interval,
            )
            chunk_days = int(meta.get("max_period_days") or 180)
            print(f"Using EOD chunk_days={chunk_days}. Probe meta={meta}")
    else:
        chunk_days = args.chunk_days if args.chunk_days > 0 else 7

    frames: list[pd.DataFrame] = []
    meta_by_window: list[dict[str, Any]] = []
    cur = start_d
    while cur <= end_d:
        window_end = min(end_d, cur + timedelta(days=chunk_days - 1))
        s = cur.strftime("%Y-%m-%d")
        e = window_end.strftime("%Y-%m-%d")
        print(f"Fetching {args.symbol} {args.mode}: {s} .. {e}")
        df, meta = fetch_market_window(args.mode, args.symbol, s, e, token, args.interval)
        print(f"  rows={len(df):,} meta={meta}")
        meta_by_window.append({"start": s, "end": e, "rows": len(df), "meta": meta})
        if not df.empty:
            frames.append(df)
        elif not args.continue_on_empty:
            print("Stopping on empty window. Use --continue-on-empty to keep going.")
            break
        cur = window_end + timedelta(days=1)
        if cur <= end_d and args.sleep > 0:
            time.sleep(args.sleep)

    if not frames:
        print("No market data downloaded.")
        return

    out = combine_windows(frames)
    descriptor = "eod" if args.mode == "eod" else f"intraday_{args.interval}"
    outdir = Path(args.outdir) if args.outdir else DEFAULT_DATA_DIR / "stock_data" / args.symbol
    stem = f"{args.symbol}_{descriptor}_{args.start}_to_{args.end}"
    write_frame(out, outdir, stem, csv=args.csv, parquet=args.parquet, meta={"windows": meta_by_window})


def fetch_news_window(
    url: str,
    symbols: str,
    start: str,
    end: str,
    token: str,
    language: str | None,
    additional_params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    params: dict[str, Any] = {
        "api_token": token,
        "symbols": symbols,
        "date_from": start,
        "date_to": end,
    }
    if language:
        params["language"] = language
    params.update(additional_params)
    payload = request_json(url, params, timeout=90)
    df, meta = payload_to_frame(payload)
    return normalize_date_column(df), meta


def parse_kv_params(values: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected key=value for --param, got: {value}")
        key, val = value.split("=", 1)
        out[key] = val
    return out


def run_news(args: argparse.Namespace) -> None:
    token = get_api_token()
    start_d = to_date(args.start)
    end_d = to_date(args.end)
    if end_d < start_d:
        raise ValueError("--end must be >= --start")
    if args.chunk_days <= 0:
        raise ValueError("--chunk-days must be > 0")

    additional_params = parse_kv_params(args.param)
    frames: list[pd.DataFrame] = []
    meta_by_window: list[dict[str, Any]] = []
    cur = start_d
    while cur <= end_d:
        window_end = min(end_d, cur + timedelta(days=args.chunk_days - 1))
        s = cur.strftime("%Y-%m-%d")
        e = window_end.strftime("%Y-%m-%d")
        print(f"Fetching news for {args.symbols}: {s} .. {e}")
        df, meta = fetch_news_window(args.url, args.symbols, s, e, token, args.language, additional_params)
        print(f"  rows={len(df):,} meta={meta}")
        meta_by_window.append({"start": s, "end": e, "rows": len(df), "meta": meta})
        if not df.empty:
            if "symbols" not in df.columns and "symbol_target" not in df.columns:
                df["symbol_target"] = args.symbols
            frames.append(df)
        elif not args.continue_on_empty:
            print("Stopping on empty window. Use --continue-on-empty to keep going.")
            break
        cur = window_end + timedelta(days=1)
        if cur <= end_d and args.sleep > 0:
            time.sleep(args.sleep)

    if not frames:
        print("No news data downloaded.")
        return

    out = combine_windows(frames)
    safe_symbols = args.symbols.replace(",", "_")
    outdir = Path(args.outdir) if args.outdir else DEFAULT_DATA_DIR / "news_headlines"
    stem = f"{safe_symbols}_news_{args.start}_to_{args.end}"
    write_frame(out, outdir, stem, csv=args.csv, parquet=args.parquet, meta={"windows": meta_by_window})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch StockData.org raw market data and news.")
    sub = parser.add_subparsers(dest="command", required=True)

    market = sub.add_parser("market", help="Fetch EOD or intraday market data.")
    market.add_argument("--mode", choices=("eod", "intraday"), default="eod")
    market.add_argument("--symbol", default="NVDA")
    market.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    market.add_argument("--end", default=str(date.today()), help="End date YYYY-MM-DD")
    market.add_argument("--outdir", default=None)
    market.add_argument("--csv", action="store_true", help="Also write CSV")
    market.add_argument("--no-parquet", dest="parquet", action="store_false", help="Do not write Parquet")
    market.set_defaults(parquet=True)
    market.add_argument("--sleep", type=float, default=0.5)
    market.add_argument("--continue-on-empty", action="store_true")
    market.add_argument("--chunk-days", type=int, default=0, help="0 means use API EOD limit or 7 days for intraday")
    market.add_argument("--interval", default="minute", help="Intraday interval")
    market.set_defaults(func=run_market)

    news = sub.add_parser("news", help="Fetch news headlines.")
    news.add_argument("--symbols", default="NVDA", help="Comma-separated ticker symbols")
    news.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    news.add_argument("--end", default=str(date.today()), help="End date YYYY-MM-DD")
    news.add_argument("--outdir", default=None)
    news.add_argument("--csv", action="store_true", help="Also write CSV")
    news.add_argument("--no-parquet", dest="parquet", action="store_false", help="Do not write Parquet")
    news.set_defaults(parquet=True)
    news.add_argument("--sleep", type=float, default=0.5)
    news.add_argument("--continue-on-empty", action="store_true")
    news.add_argument("--chunk-days", type=int, default=30)
    news.add_argument("--language", default="en")
    news.add_argument("--url", default=NEWS_API_URL, help="News endpoint URL; override if StockData changes it")
    news.add_argument("--param", action="append", help="Additional API query parameter as key=value")
    news.set_defaults(func=run_news)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
