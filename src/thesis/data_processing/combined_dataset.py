import pandas as pd
import os

# ================================
# INPUT FILES
# ================================
NVDA_PATH = "/home/zammorak/thesis/data/processed/NVDA_eod_processed.csv"
SPY_PATH = "/home/zammorak/thesis/data/processed/SPY_eod_processed.csv"
SOXX_PATH = "/home/zammorak/thesis/data/processed/SOXX_eod_processed.csv"
IEF_PATH = "/home/zammorak/thesis/data/processed/IEF_eod_processed.csv"
SENT_PATH = "/home/zammorak/thesis/data/processed/news_daily_sentiment.csv"
TRENDS_PATH = "/home/zammorak/thesis/data/processed/nvidia_trends_processed.csv"

# ================================
# OUTPUT
# ================================
OUTPUT_DIR = "/home/zammorak/thesis/data/model_feed"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "model_dataset.csv")


def build_model_dataset():

    # --------------------------
    # Load data
    # --------------------------
    nvda = pd.read_csv(NVDA_PATH)
    spy = pd.read_csv(SPY_PATH)
    soxx = pd.read_csv(SOXX_PATH)
    ief = pd.read_csv(IEF_PATH)
    sent = pd.read_csv(SENT_PATH)
    trends = pd.read_csv(TRENDS_PATH)

    # --------------------------
    # Standardize dates
    # --------------------------
    nvda["date"] = pd.to_datetime(nvda["date"])
    spy["date"] = pd.to_datetime(spy["date"])
    soxx["date"] = pd.to_datetime(soxx["date"])
    ief["date"] = pd.to_datetime(ief["date"])
    sent["trading_date"] = pd.to_datetime(sent["trading_date"])
    trends["date"] = pd.to_datetime(trends["date"])

    # --------------------------
    # Keep only needed columns
    # --------------------------
    spy = spy[["date", "log_return"]].rename(columns={"log_return": "spy_return"})
    soxx = soxx[["date", "log_return"]].rename(columns={"log_return": "soxx_return"})
    ief = ief[["date", "log_return"]].rename(columns={"log_return": "ief_return"})

    # --------------------------
    # Merge macro factors
    # --------------------------
    df = nvda.merge(spy, on="date", how="left")
    df = df.merge(soxx, on="date", how="left")
    df = df.merge(ief, on="date", how="left")

    # --------------------------
    # Merge sentiment
    # --------------------------
    df = df.merge(
        sent,
        left_on="date",
        right_on="trading_date",
        how="left"
    )

    df = df.drop(columns=["trading_date"], errors="ignore")

    # --------------------------
    # Merge trends
    # --------------------------
    df = df.merge(trends, on="date", how="left")

    # --------------------------
    # Forward fill sentiment & trends
    # (optional but common practice)
    # --------------------------
    df = df.sort_values("date")
    df[[
        "avg_sentiment",
        "median_sentiment",
        "sentiment_std",
        "news_count",
        "positive_ratio",
        "negative_ratio",
        "trends_raw",
        "trends_7d_ma",
        "trends_momentum_7d",
        "trends_zscore_30d",
        "trends_spike"
    ]] = df[[
        "avg_sentiment",
        "median_sentiment",
        "sentiment_std",
        "news_count",
        "positive_ratio",
        "negative_ratio",
        "trends_raw",
        "trends_7d_ma",
        "trends_momentum_7d",
        "trends_zscore_30d",
        "trends_spike"
    ]].fillna(method="ffill")

    # --------------------------
    # Drop remaining NaNs
    # --------------------------
    df = df.dropna().reset_index(drop=True)

    # --------------------------
    # Save final dataset
    # --------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    return df


if __name__ == "__main__":
    final_df = build_model_dataset()

    print("✅ Model dataset created successfully")
    print("Rows:", len(final_df))
    print("Columns:", len(final_df.columns))
    print("Saved to:", OUTPUT_PATH)