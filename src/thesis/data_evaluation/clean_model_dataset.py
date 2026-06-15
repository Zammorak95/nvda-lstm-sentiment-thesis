#!/usr/bin/env python3

import pandas as pd
import os

INPUT_PATH = "/home/zammorak/thesis/data/model_feed/model_dataset.csv"
OUTPUT_PATH = "/home/zammorak/thesis/data/model_feed/model_dataset_clean.csv"

print("Loading dataset...")
df = pd.read_csv(INPUT_PATH)

# --- Features to keep (clean feature set) ---
FEATURES = [
    "log_return",
    "overnight_return",

    "momentum_5d",
    "momentum_20d",
    "volatility_20d",

    "volume_change",
    "volume_20d_avg",

    "avg_sentiment",
    "sentiment_std",
    "news_count",

    "spy_return",
    "soxx_return",
    "ief_return",

    "trends_zscore_30d",
    "trends_momentum_7d",
    "trends_spike"
]

TARGETS = [
    "target_direction",
    "target_next_return"
]

META = ["date"]

# Keep only existing columns
keep_cols = [c for c in META + FEATURES + TARGETS if c in df.columns]

clean_df = df[keep_cols].copy()

print("\nColumns kept:")
for c in clean_df.columns:
    print("  ", c)

print("\nOriginal shape:", df.shape)
print("Clean shape:", clean_df.shape)

# Save dataset
clean_df.to_csv(OUTPUT_PATH, index=False)

print("\nClean dataset saved to:")
print(OUTPUT_PATH)