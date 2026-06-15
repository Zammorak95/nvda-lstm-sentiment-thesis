#!/usr/bin/env python3

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

DATA = "/home/zammorak/thesis/data/model_feed/model_dataset.csv"

df = pd.read_csv(DATA)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# remove non-features
drop_cols = ["date", "target_direction", "target_next_return"]
features = [c for c in df.columns if c not in drop_cols]

X = df[features]

corr = X.corr()

# find highly correlated pairs
threshold = 0.85
high_corr = []

for i in range(len(corr.columns)):
    for j in range(i):
        if abs(corr.iloc[i, j]) > threshold:
            high_corr.append(
                (corr.columns[i], corr.columns[j], corr.iloc[i, j])
            )

print("\nHighly correlated features (>0.85):\n")
for f1, f2, c in high_corr:
    print(f"{f1:25s} {f2:25s} corr={c:.3f}")

# heatmap
plt.figure(figsize=(14,10))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

print("\nCorrelation with target_direction:\n")

target_corr = df[features + ["target_direction"]].corr()["target_direction"]
target_corr = target_corr.drop("target_direction").sort_values()

print(target_corr)

from statsmodels.stats.outliers_influence import variance_inflation_factor

print("\nVariance Inflation Factor (VIF):\n")

vif = pd.DataFrame()
vif["feature"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif.sort_values("VIF", ascending=False))

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=500)
rf.fit(X, df["target_direction"])

importance = pd.Series(rf.feature_importances_, index=X.columns)
print(importance.sort_values(ascending=False))