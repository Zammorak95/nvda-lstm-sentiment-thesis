import pandas as pd
import os

INPUT_PATH = "/home/zammorak/thesis/data/interim/news_headlines_master.csv"
OUTPUT_PATH = "/home/zammorak/thesis/data/processed/news_headlines_clean.csv"


def clean_news_dataset(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    # --- Standardize column names ---
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # --- Keep only required columns ---
    required_cols = [
        "symbol_target",
        "published_at",
        "title",
        "description",
        "keywords",
        "source",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df[required_cols].copy()

    # --- Clean text columns ---
    text_cols = ["title", "description", "keywords", "source"]
    for col in text_cols:
        # keep NA as NA (don't turn into "nan" strings)
        df[col] = df[col].where(df[col].notna(), None)
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": None, "nan": None, "None": None})

    # Drop rows without title
    df = df[df["title"].notna()]

    # --- Convert published_at to proper datetime (UTC) ---
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df = df[df["published_at"].notna()]

    # =========================================================
    # TRADING MODEL ALIGNMENT (US equities, 16:00 ET close)
    # 1) Convert to US/Eastern
    # 2) Remove weekend news
    # 3) If published after 16:00 ET -> map to next trading day
    # 4) If mapped day falls on weekend -> push to Monday
    # =========================================================

    df["published_et"] = df["published_at"].dt.tz_convert("America/New_York")
    df["weekday_et"] = df["published_et"].dt.weekday  # 0=Mon ... 6=Sun

    # Remove weekend news (Sat=5, Sun=6)
    df = df[df["weekday_et"] < 5].copy()

    # Trading date starts as the ET calendar date
    df["trading_date"] = df["published_et"].dt.floor("D")

    # After-close rule (>= 16:00 ET -> next day)
    market_close = 16
    after_close_mask = df["published_et"].dt.hour >= market_close
    df.loc[after_close_mask, "trading_date"] = df.loc[after_close_mask, "trading_date"] + pd.Timedelta(days=1)

    # If shifted onto weekend, push to Monday
    # Saturday -> +2 days, Sunday -> +1 day
    saturday_mask = df["trading_date"].dt.weekday == 5
    sunday_mask = df["trading_date"].dt.weekday == 6
    df.loc[saturday_mask, "trading_date"] = df.loc[saturday_mask, "trading_date"] + pd.Timedelta(days=2)
    df.loc[sunday_mask, "trading_date"] = df.loc[sunday_mask, "trading_date"] + pd.Timedelta(days=1)

    # Make trading_date a simple date (good for merging with daily OHLC)
    df["trading_date"] = df["trading_date"].dt.date

    # --- Create useful time features (based on TRADING_DATE) ---
    # Keep the original published_at too (timestamp)
    df["date"] = df["published_at"].dt.date
    df["year"] = df["published_at"].dt.year
    df["month"] = df["published_at"].dt.month
    df["day"] = df["published_at"].dt.day
    df["hour"] = df["published_at"].dt.hour
    df["weekday"] = df["published_at"].dt.weekday  # weekday in UTC
    df["is_weekend"] = df["weekday"] >= 5

    # Sort chronologically
    df = df.sort_values("published_at").reset_index(drop=True)

    # Drop helper column (optional)
    df = df.drop(columns=["weekday_et"])

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    cleaned_df = clean_news_dataset(INPUT_PATH, OUTPUT_PATH)

    print("✅ Clean + trading-aligned dataset created")
    print("Rows:", len(cleaned_df))
    print("Columns:", cleaned_df.columns.tolist())
    print("Saved to:", OUTPUT_PATH)