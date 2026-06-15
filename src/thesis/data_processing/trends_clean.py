import pandas as pd
import numpy as np
import os

INPUT_PATH = "/home/zammorak/thesis/data/interim/nvidia_trends_daily_consistent.csv"
OUTPUT_PATH = "/home/zammorak/thesis/data/processed/nvidia_trends_processed.csv"


def clean_and_process_trends(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    # -------------------------------------------------
    # 1. Standardize columns
    # -------------------------------------------------
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # Expect columns like: date, nvidia (or similar)
    if "date" not in df.columns:
        raise ValueError("No 'date' column found in trends data.")

    # Detect trends value column automatically
    value_cols = [c for c in df.columns if c != "date"]
    if len(value_cols) != 1:
        raise ValueError("Unexpected trends structure. Should contain one value column.")

    trend_col = value_cols[0]

    # -------------------------------------------------
    # 2. Clean date
    # -------------------------------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]
    df = df.sort_values("date").reset_index(drop=True)

    # -------------------------------------------------
    # 3. Rename trend column
    # -------------------------------------------------
    df = df.rename(columns={trend_col: "trends_raw"})

    # Ensure numeric
    df["trends_raw"] = pd.to_numeric(df["trends_raw"], errors="coerce")

    # -------------------------------------------------
    # 4. Feature Engineering (Important)
    # -------------------------------------------------

    # Smooth trend (reduces noise)
    df["trends_7d_ma"] = df["trends_raw"].rolling(7).mean()

    # Trend momentum
    df["trends_momentum_7d"] = df["trends_raw"].pct_change(7)

    # Abnormal attention (Z-score, 30-day rolling)
    rolling_mean = df["trends_raw"].rolling(30).mean()
    rolling_std = df["trends_raw"].rolling(30).std()

    df["trends_zscore_30d"] = (df["trends_raw"] - rolling_mean) / rolling_std

    # Spike indicator (attention shock)
    df["trends_spike"] = (df["trends_zscore_30d"] > 2).astype(int)

    # -------------------------------------------------
    # 5. Drop initial NaNs from rolling windows
    # -------------------------------------------------
    df = df.dropna().reset_index(drop=True)

    # -------------------------------------------------
    # 6. Save
    # -------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    trends_df = clean_and_process_trends(INPUT_PATH, OUTPUT_PATH)

    print("✅ Trends processed successfully")
    print("Rows:", len(trends_df))
    print("Columns:", trends_df.columns.tolist())
    print("Saved to:", OUTPUT_PATH)