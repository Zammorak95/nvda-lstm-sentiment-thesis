from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pytrends.request import TrendReq


# -------------------------------------------------
# Settings
# -------------------------------------------------
KEYWORD = "AMD stock"
START_DATE = "2016-01-01"
END_DATE = "2026-03-01"

OUTDIR = Path("/home/zammorak/thesis/data/interim")
OUTDIR.mkdir(parents=True, exist_ok=True)

RAW_OUTDIR = Path("/home/zammorak/thesis/data/raw/trends_amd_pytrends")
RAW_OUTDIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTDIR / "amd_trends_daily_consistent.csv"

# Worldwide, web search
GEO = ""
GPROP = ""

# Google Trends usually returns daily data for shorter windows.
# 90-day chunks are a safe choice.
CHUNK_DAYS = 90
SLEEP_SECONDS = 12


def fetch_interest(pytrends: TrendReq, keyword: str, start: str, end: str) -> pd.DataFrame:
    timeframe = f"{start} {end}"
    pytrends.build_payload(
        kw_list=[keyword],
        cat=0,
        timeframe=timeframe,
        geo=GEO,
        gprop=GPROP,
    )

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


def date_chunks(start_date: str, end_date: str, chunk_days: int):
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    cur = start
    while cur <= end:
        chunk_end = min(end, cur + timedelta(days=chunk_days - 1))
        yield cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
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

    if ratios.empty:
        return 1.0

    return float(ratios.median())


def main() -> None:
    pytrends = TrendReq(
        hl="en-US",
        tz=360,
        timeout=(10, 25),
        retries=3,
        backoff_factor=0.5,
    )

    print(f"Fetching monthly/global reference for: {KEYWORD}")
    reference = fetch_interest(pytrends, KEYWORD, START_DATE, END_DATE)
    reference.to_csv(RAW_OUTDIR / "amd_stock_reference_full_period.csv", index=False)

    chunks = []

    for i, (start, end) in enumerate(date_chunks(START_DATE, END_DATE, CHUNK_DAYS), start=1):
        print(f"Fetching daily chunk {i:02d}: {start} -> {end}")

        try:
            chunk = fetch_interest(pytrends, KEYWORD, start, end)
        except Exception as exc:
            print(f"Failed chunk {start} -> {end}: {exc}")
            raise

        chunk_path = RAW_OUTDIR / f"amd_stock_chunk_{i:02d}_{start}_to_{end}.csv"
        chunk.to_csv(chunk_path, index=False)

        scale = compute_scale(chunk, reference)
        chunk["value"] = chunk["value"] * scale
        chunk["chunk"] = i
        chunk["scale_to_reference"] = scale

        chunks.append(chunk)

        time.sleep(SLEEP_SECONDS)

    if not chunks:
        raise RuntimeError("No chunks were fetched.")

    # Chain-link chunk boundaries for smoother continuity
    scaled_chunks = chunks

    for i in range(1, len(scaled_chunks)):
        prev = scaled_chunks[i - 1]
        curr = scaled_chunks[i]

        prev_last_value = float(prev.loc[prev["date"] == prev["date"].max(), "value"].iloc[0])
        curr_first_value = float(curr.loc[curr["date"] == curr["date"].min(), "value"].iloc[0])

        if abs(curr_first_value) > 1e-9:
            factor = prev_last_value / curr_first_value
            scaled_chunks[i]["value"] = scaled_chunks[i]["value"] * factor
            scaled_chunks[i]["chain_link_factor"] = factor
        else:
            scaled_chunks[i]["chain_link_factor"] = 1.0

    daily = pd.concat(scaled_chunks, ignore_index=True)
    daily = daily.sort_values("date")

    # Resolve overlaps by averaging
    daily = daily.groupby("date", as_index=False)["value"].mean()

    # Make continuous daily index
    daily = daily.set_index("date").asfreq("D")
    daily["value"] = daily["value"].interpolate(limit_direction="both")
    daily = daily.reset_index()

    daily = daily.rename(columns={"value": "amd_stock_trends"})
    daily.to_csv(OUTPUT_PATH, index=False)

    print()
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(daily):,}")
    print(f"Range: {daily['date'].min().date()} -> {daily['date'].max().date()}")
    print(daily.head())
    print(daily.tail())


if __name__ == "__main__":
    main()

