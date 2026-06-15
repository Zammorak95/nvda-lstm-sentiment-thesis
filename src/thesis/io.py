# src/thesis/io.py
from pathlib import Path
import pandas as pd
from typing import Optional
from thesis.paths import RAW

__all__ = ["load_data"]  # <- optional, but makes exports explicit

def load_data(symbol: str, interval: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    folder = RAW / interval / symbol
    if not folder.exists():
        raise FileNotFoundError(f"No data found at {folder}")

    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {folder}")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    # decide time column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        tcol = "date"
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        tcol = "timestamp"
    else:
        raise ValueError("No 'date' or 'timestamp' column found")

    if start:
        df = df[df[tcol] >= pd.to_datetime(start)]
    if end:
        df = df[df[tcol] <= pd.to_datetime(end)]
    return df
