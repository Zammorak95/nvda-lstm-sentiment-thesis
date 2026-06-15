#!/usr/bin/env python3
"""Clean and analyze a model dataset for feature selection.

This combines:
- clean_model_dataset.py: keep a curated clean feature set and save it.
- feature_correlation_analysis.py: correlation matrix, high-correlation pairs,
  target correlations, VIF, and RandomForest feature importance.

Examples:
  # Clean only
  python model_dataset_clean_and_analyze.py clean --input data/model_dataset.csv --output data/model_dataset_clean.csv

  # Analyze only
  python model_dataset_clean_and_analyze.py analyze --input data/model_dataset.csv --plot

  # Clean, then analyze the cleaned output
  python model_dataset_clean_and_analyze.py all --input data/model_dataset.csv --output data/model_dataset_clean.csv --plot
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_INPUT = "/home/zammorak/thesis/data/model_feed/model_dataset.csv"
DEFAULT_OUTPUT = "/home/zammorak/thesis/data/model_feed/model_dataset_clean.csv"

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
    "trends_spike",
]

TARGETS = ["target_direction", "target_next_return"]
META = ["date"]


def load_dataset(path: str | Path) -> pd.DataFrame:
    print(f"Loading dataset: {path}")
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
    return df


def clean_dataset(df: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    """Keep curated metadata, feature, and target columns that exist in the input."""
    keep_cols = [c for c in META + FEATURES + TARGETS if c in df.columns]
    clean_df = df[keep_cols].copy()

    print("\nColumns kept:")
    for col in clean_df.columns:
        print(f"  {col}")

    print(f"\nOriginal shape: {df.shape}")
    print(f"Clean shape:    {clean_df.shape}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_path, index=False)
    print(f"\nClean dataset saved to:\n{output_path}")

    return clean_df


def feature_columns(df: pd.DataFrame, extra_drop: Iterable[str] = ()) -> list[str]:
    drop_cols = set(META + TARGETS + list(extra_drop))
    return [c for c in df.columns if c not in drop_cols]


def numeric_feature_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = df[features].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    # VIF and RandomForest cannot handle NaN/inf. Median fill keeps rows without lookahead.
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.fillna(X.median(numeric_only=True))


def print_high_correlations(corr: pd.DataFrame, threshold: float) -> None:
    high_corr: list[tuple[str, str, float]] = []
    for i in range(len(corr.columns)):
        for j in range(i):
            value = corr.iloc[i, j]
            if pd.notna(value) and abs(value) > threshold:
                high_corr.append((corr.columns[i], corr.columns[j], value))

    print(f"\nHighly correlated features (>{threshold} absolute correlation):\n")
    if not high_corr:
        print("  None")
        return

    for f1, f2, value in high_corr:
        print(f"{f1:25s} {f2:25s} corr={value:.3f}")


def plot_heatmap(corr: pd.DataFrame, output_path: str | Path | None = None, show: bool = False) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)
        print(f"\nCorrelation heatmap saved to:\n{output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def analyze_dataset(
    df: pd.DataFrame,
    threshold: float = 0.85,
    plot: bool = False,
    heatmap_output: str | Path | None = None,
    random_state: int = 42,
) -> None:
    features = feature_columns(df)
    X = numeric_feature_matrix(df, features)

    corr = X.corr()
    print_high_correlations(corr, threshold)

    if plot or heatmap_output:
        plot_heatmap(corr, output_path=heatmap_output, show=plot)

    if "target_direction" in df.columns:
        print("\nCorrelation with target_direction:\n")
        target_data = pd.concat([X, pd.to_numeric(df["target_direction"], errors="coerce")], axis=1)
        target_corr = target_data.corr()["target_direction"].drop("target_direction").sort_values()
        print(target_corr)
    else:
        print("\nSkipping target correlation: target_direction column not found.")

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        print("\nVariance Inflation Factor (VIF):\n")
        vif = pd.DataFrame()
        vif["feature"] = X.columns
        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        print(vif.sort_values("VIF", ascending=False))
    except Exception as exc:  # noqa: BLE001
        print(f"\nSkipping VIF calculation: {exc}")

    if "target_direction" in df.columns:
        try:
            from sklearn.ensemble import RandomForestClassifier

            y = pd.to_numeric(df["target_direction"], errors="coerce")
            valid = y.notna()
            rf = RandomForestClassifier(n_estimators=500, random_state=random_state)
            rf.fit(X.loc[valid], y.loc[valid].astype(int))

            print("\nRandomForest feature importance:\n")
            importance = pd.Series(rf.feature_importances_, index=X.columns)
            print(importance.sort_values(ascending=False))
        except Exception as exc:  # noqa: BLE001
            print(f"\nSkipping RandomForest importance: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and/or analyze a model dataset.")
    parser.add_argument(
        "command",
        choices=["clean", "analyze", "all"],
        help="clean: save curated columns; analyze: run diagnostics; all: clean then analyze cleaned data.",
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"Input CSV path (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Clean CSV output path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--threshold", type=float, default=0.85, help="High-correlation threshold (default: 0.85)")
    parser.add_argument("--plot", action="store_true", help="Display the correlation heatmap interactively")
    parser.add_argument("--heatmap-output", default=None, help="Optional path to save the correlation heatmap image")
    parser.add_argument("--random-state", type=int, default=42, help="RandomForest random_state (default: 42)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(args.input)

    if args.command == "clean":
        clean_dataset(df, args.output)
    elif args.command == "analyze":
        analyze_dataset(
            df,
            threshold=args.threshold,
            plot=args.plot,
            heatmap_output=args.heatmap_output,
            random_state=args.random_state,
        )
    elif args.command == "all":
        clean_df = clean_dataset(df, args.output)
        analyze_dataset(
            clean_df,
            threshold=args.threshold,
            plot=args.plot,
            heatmap_output=args.heatmap_output,
            random_state=args.random_state,
        )


if __name__ == "__main__":
    main()
