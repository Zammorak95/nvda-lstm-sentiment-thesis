import pandas as pd
import numpy as np

DATA_PATH = "/home/zammorak/thesis/data/model_feed/model_dataset.csv"

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

print("===================================")
print("BASIC INFO")
print("===================================")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("\nDate Range:", df["date"].min(), "→", df["date"].max())


# =====================================================
# 1️⃣ Check duplicate dates
# =====================================================
duplicates = df["date"].duplicated().sum()
print("\nDuplicate dates:", duplicates)


# =====================================================
# 2️⃣ Check missing dates (trading gaps)
# =====================================================
df = df.sort_values("date").reset_index(drop=True)

date_diffs = df["date"].diff().dt.days
large_gaps = date_diffs[date_diffs > 3]  # >3 catches abnormal gaps

print("\nPotential abnormal gaps (>3 days):")
print(large_gaps)

print("Max gap (days):", date_diffs.max())


# =====================================================
# 3️⃣ Check missing values per column
# =====================================================
print("\n===================================")
print("MISSING VALUES PER COLUMN")
print("===================================")

missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing) == 0:
    print("No missing values detected ✅")
else:
    print(missing)


# =====================================================
# 4️⃣ Check % missing
# =====================================================
missing_pct = (df.isna().mean() * 100).round(2)
missing_pct = missing_pct[missing_pct > 0]

if len(missing_pct) > 0:
    print("\nMissing percentage (%):")
    print(missing_pct)


# =====================================================
# 5️⃣ Check zero variance columns
# =====================================================
zero_var = df.nunique()
zero_var = zero_var[zero_var <= 1]

if len(zero_var) > 0:
    print("\nColumns with zero variance:")
    print(zero_var)


# =====================================================
# 6️⃣ Quick sanity checks on key columns
# =====================================================
print("\n===================================")
print("SANITY CHECKS")
print("===================================")

print("Return mean:", df["log_return"].mean())
print("Return std:", df["log_return"].std())
print("Sentiment mean:", df["avg_sentiment"].mean())
print("Trends z-score mean:", df["trends_zscore_30d"].mean())

print("\nAudit completed.")


corr = df.corr(numeric_only=True)
print(corr["target_next_return"].sort_values(ascending=False))

df["trends_z_lag1"] = df["trends_zscore_30d"].shift(1)
df["trends_z_lag2"] = df["trends_zscore_30d"].shift(2)

df[["trends_z_lag1", "trends_z_lag2"]].corrwith(df["target_next_return"])

df["sent_x_trend"] = df["avg_sentiment"] * df["trends_zscore_30d"]

df["volatility_20d"].shift(-1)

df.groupby(df["trends_spike"])["target_next_return"].mean()