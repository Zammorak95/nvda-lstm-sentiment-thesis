import pandas as pd
import numpy as np
import os

INPUT_PATH = "/home/zammorak/thesis/data/raw/macro_stock_data/SPY/SPY_eod_chunked_2019-03-01_to_2026-03-01.csv"
OUTPUT_PATH = "/home/zammorak/thesis/data/processed/SPY_eod_processed.csv"


def clean_and_process_eod(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    # -------------------------------------------------
    # 1. Standardize column names
    # -------------------------------------------------
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    required_price_cols = ["date", "open", "high", "low", "close"]
    missing = [c for c in required_price_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # -------------------------------------------------
    # 2. Clean date column
    # -------------------------------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]

    # Remove duplicates
    df = df.drop_duplicates(subset=["date"])

    # Sort chronologically
    df = df.sort_values("date").reset_index(drop=True)

    # -------------------------------------------------
    # 3. Feature Engineering
    # -------------------------------------------------

    # Simple return
    df["return"] = df["close"].pct_change()

    # Log return (preferred for modeling)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Intraday return
    df["intraday_return"] = (df["close"] - df["open"]) / df["open"]

    # Overnight return
    df["overnight_return"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

    # Rolling volatility (20 trading days ≈ 1 month)
    df["volatility_20d"] = df["log_return"].rolling(20).std()

    # Momentum factors
    df["momentum_5d"] = df["log_return"].rolling(5).mean()
    df["momentum_20d"] = df["log_return"].rolling(20).mean()

    # Volume features (if available)
    if "volume" in df.columns:
        df["volume_change"] = df["volume"].pct_change()
        df["volume_20d_avg"] = df["volume"].rolling(20).mean()

    # -------------------------------------------------
    # 4. Targets (VERY IMPORTANT)
    # -------------------------------------------------

    # Next-day log return (regression target)
    df["target_next_return"] = df["log_return"].shift(-1)

    # Direction classification target
    df["target_direction"] = (df["target_next_return"] > 0).astype(int)

    # -------------------------------------------------
    # 5. Clean NaNs from rolling windows
    # -------------------------------------------------
    df = df.dropna().reset_index(drop=True)

    # -------------------------------------------------
    # 6. Save processed dataset
    # -------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    processed_df = clean_and_process_eod(INPUT_PATH, OUTPUT_PATH)

    print("✅ NVDA EOD processed successfully")
    print("Rows:", len(processed_df))
    print("Columns:", processed_df.columns.tolist())
    print("Saved to:", OUTPUT_PATH)