#!/usr/bin/env python3
"""
Fast, type-checker-friendly expansion of a nested 'data' column from a Parquet file.
- Loads the raw parquet
- Expands dicts from the 'data' column (using from_records for Pylance friendliness)
- Parses date (UTC) and optionally converts to Europe/Amsterdam
- Sets a DatetimeIndex
- Saves an expanded parquet (creating directories if needed)
- Prints quick sanity checks + timing breakdown
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# --- settings ---
INFILE: str = "/home/zammorak/thesis-starter/thesis/data/raw/minute/NVDA/NVDA_2025-09-26_to_2025-10-02.parquet"
OUTFILE: str = "/home/zammorak/thesis-starter/thesis/data/processed/minute/NVDA/NVDA_2025-09-26_to_2025-10-02_expanded.parquet"
CONVERT_TO_AMS: bool = False  # set True if you want local Amsterdam time


def expand_data_column(df: pd.DataFrame, data_col: str = "data") -> pd.DataFrame:
    """Expand a column of dicts into flat columns (type-checker friendly)."""
    if data_col not in df.columns:
        return df
    # Convert Series[dict] -> list[dict] to satisfy static type checkers
    records: List[Dict[str, Any]] = df[data_col].tolist()
    expanded = pd.DataFrame.from_records(records)
    return df.drop(columns=[data_col]).join(expanded)


def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce common OHLCV + flags to sensible dtypes, if present."""
    float_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
    int_cols = [c for c in ("volume",) if c in df.columns]
    bool_cols = [c for c in ("is_extended_hours",) if c in df.columns]

    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in int_cols:
        # Nullable Int64 handles missing values gracefully
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    for c in bool_cols:
        if df[c].dtype != "bool" and df[c].dtype.name != "boolean":
            # 'boolean' is pandas' nullable boolean
            df[c] = df[c].astype("boolean")

    return df


def parse_and_index_date(
    df: pd.DataFrame, date_col: str = "date", to_ams: bool = False
) -> pd.DataFrame:
    """Parse ISO-8601 date strings to UTC, optional tz convert, and index."""
    if date_col not in df.columns:
        raise KeyError(f"Expected '{date_col}' in DataFrame columns.")
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    if to_ams:
        df[date_col] = df[date_col].dt.tz_convert("Europe/Amsterdam")
    df = df.set_index(date_col).sort_index()
    return df


def main(
    infile: str = INFILE, outfile: str = OUTFILE, convert_to_ams: bool = CONVERT_TO_AMS
) -> None:
    t0 = time.perf_counter()

    # 1) Load
    df = pd.read_parquet(infile)
    t1 = time.perf_counter()

    # 2) Expand nested dicts (vectorized, type-checker friendly)
    df = expand_data_column(df, data_col="data")
    t2 = time.perf_counter()

    # 3) Parse date and set index
    df = parse_and_index_date(df, date_col="date", to_ams=convert_to_ams)
    t3 = time.perf_counter()

    # 4) Dtype cleanup
    df = coerce_dtypes(df)
    t4 = time.perf_counter()

    # 5) Ensure output directory exists, then save (keep time index)
    out_path = Path(outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=True)
    t5 = time.perf_counter()

    # 6) Sanity + timings
    print(df.head(5))
    print()
    print(df.info())

    print("\n— Timing —")
    print(f"Load parquet:        {t1 - t0:6.2f}s")
    print(f"Expand (from_records):{t2 - t1:6.2f}s")
    print(f"Datetime/index:      {t3 - t2:6.2f}s")
    print(f"Dtype cleanup:       {t4 - t3:6.2f}s")
    print(f"Save expanded:       {t5 - t4:6.2f}s")
    print(f"Total:               {t5 - t0:6.2f}s")


if __name__ == "__main__":
    main()
