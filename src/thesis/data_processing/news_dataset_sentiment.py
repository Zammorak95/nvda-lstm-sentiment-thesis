import pandas as pd
import numpy as np
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer

INPUT_PATH = "/home/zammorak/thesis/data/processed/news_headlines_clean.csv"
OUTPUT_PATH = "/home/zammorak/thesis/data/processed/news_daily_sentiment.csv"


def build_daily_sentiment(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    # Ensure datetime
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)

    if "trading_date" not in df.columns:
        raise ValueError("trading_date column missing. Run trading alignment first.")

    # --- Initialize VADER ---
    sid = SentimentIntensityAnalyzer()

    # --- Combine title + description for stronger signal ---
    df["text_for_sentiment"] = (
        df["title"].fillna("") + ". " + df["description"].fillna("")
    )

    # --- Compute sentiment (compound score between -1 and 1) ---
    df["sentiment"] = df["text_for_sentiment"].apply(
        lambda x: sid.polarity_scores(x)["compound"]
    )

    # --- Aggregate to trading_date ---
    daily = df.groupby("trading_date").agg(
        avg_sentiment=("sentiment", "mean"),
        median_sentiment=("sentiment", "median"),
        sentiment_std=("sentiment", "std"),
        news_count=("sentiment", "count"),
        positive_ratio=("sentiment", lambda x: np.mean(x > 0)),
        negative_ratio=("sentiment", lambda x: np.mean(x < 0)),
    ).reset_index()

    # Fill NaN std for single-news days
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0)

    # Sort
    daily = daily.sort_values("trading_date").reset_index(drop=True)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    daily.to_csv(output_path, index=False)

    return daily


if __name__ == "__main__":
    daily_sentiment = build_daily_sentiment(INPUT_PATH, OUTPUT_PATH)

    print("✅ Daily sentiment factor created")
    print("Rows:", len(daily_sentiment))
    print("Columns:", daily_sentiment.columns.tolist())
    print("Saved to:", OUTPUT_PATH)