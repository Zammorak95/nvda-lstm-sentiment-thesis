#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_CANDIDATES = [
    Path("data/model_feed/nvda_model_dataset_clean.csv"),
    Path("data/model_feed/model_dataset_clean.csv"),
    Path("artifacts/models/nvda_lstm_feature_ablation/datasets/full_model.csv"),
]


def find_default_dataset() -> Path:
    for p in DEFAULT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find NVDA dataset. Pass --data manually, e.g. "
        "--data data/model_feed/nvda_model_dataset_clean.csv"
    )


def save_corr_heatmap(corr: pd.DataFrame, outpath: Path, title: str) -> None:
    n = len(corr.columns)
    fig_size = max(9, n * 0.65)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(corr.values, vmin=-1, vmax=1)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation")

    # Annotate only if matrix is not too large
    if n <= 20:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to clean NVDA model dataset CSV.")
    parser.add_argument("--outdir", default="artifacts/reports/nvda_descriptives")
    args = parser.parse_args()

    data_path = Path(args.data) if args.data else find_default_dataset()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Separate targets from input features
    target_cols = [c for c in ["target_direction", "target_next_return"] if c in df.columns]
    feature_cols = [c for c in numeric_cols if c not in target_cols]

    # Descriptive statistics
    desc = df[feature_cols].describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]).T
    desc["missing"] = df[feature_cols].isna().sum()
    desc["skew"] = df[feature_cols].skew(numeric_only=True)
    desc["kurtosis"] = df[feature_cols].kurtosis(numeric_only=True)
    desc = desc.rename(columns={"50%": "median"})
    desc_rounded = desc.round(4)

    desc_rounded.to_csv(outdir / "nvda_descriptive_statistics.csv")

    # Correlation matrices
    pearson_corr = df[feature_cols].corr(method="pearson").round(4)
    spearman_corr = df[feature_cols].corr(method="spearman").round(4)

    pearson_corr.to_csv(outdir / "nvda_pearson_correlation_matrix.csv")
    spearman_corr.to_csv(outdir / "nvda_spearman_correlation_matrix.csv")

    save_corr_heatmap(
        pearson_corr,
        outdir / "nvda_pearson_correlation_heatmap.png",
        "NVDA feature correlation matrix - Pearson",
    )

    save_corr_heatmap(
        spearman_corr,
        outdir / "nvda_spearman_correlation_heatmap.png",
        "NVDA feature correlation matrix - Spearman",
    )

    # Target correlations
    target_corr_parts = []
    if "target_direction" in df.columns:
        direction_corr = df[feature_cols + ["target_direction"]].corr(method="pearson")["target_direction"]
        direction_corr = direction_corr.drop("target_direction").rename("corr_target_direction")
        target_corr_parts.append(direction_corr)

    if "target_next_return" in df.columns:
        return_corr = df[feature_cols + ["target_next_return"]].corr(method="pearson")["target_next_return"]
        return_corr = return_corr.drop("target_next_return").rename("corr_target_next_return")
        target_corr_parts.append(return_corr)

    if target_corr_parts:
        target_corr = pd.concat(target_corr_parts, axis=1)
        target_corr["max_abs_corr"] = target_corr.abs().max(axis=1)
        target_corr = target_corr.sort_values("max_abs_corr", ascending=False).round(4)
        target_corr.to_csv(outdir / "nvda_target_correlations.csv")

    # High-correlation pairs
    high_pairs = []
    corr_abs = pearson_corr.abs()
    for i, col_i in enumerate(corr_abs.columns):
        for j, col_j in enumerate(corr_abs.columns):
            if j <= i:
                continue
            val = pearson_corr.loc[col_i, col_j]
            if abs(val) >= 0.70:
                high_pairs.append({
                    "feature_1": col_i,
                    "feature_2": col_j,
                    "pearson_corr": round(float(val), 4),
                })

    high_pairs_df = pd.DataFrame(high_pairs).sort_values(
        "pearson_corr", key=lambda s: s.abs(), ascending=False
    ) if high_pairs else pd.DataFrame(columns=["feature_1", "feature_2", "pearson_corr"])

    high_pairs_df.to_csv(outdir / "nvda_high_correlation_pairs.csv", index=False)

    # Dataset summary
    summary_lines = []
    summary_lines.append("# NVDA dataset descriptive summary\n")
    summary_lines.append(f"- Dataset: `{data_path}`")
    summary_lines.append(f"- Observations: {len(df):,}")
    summary_lines.append(f"- Input features: {len(feature_cols)}")
    summary_lines.append(f"- Missing values in input features: {int(df[feature_cols].isna().sum().sum())}")

    if "date" in df.columns:
        summary_lines.append(f"- Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    if "target_direction" in df.columns:
        counts = df["target_direction"].value_counts(dropna=False).sort_index()
        up = int(counts.get(1, 0))
        down = int(counts.get(0, 0))
        summary_lines.append(f"- Class balance: {up:,} up days ({up / len(df):.2%}); {down:,} down days ({down / len(df):.2%})")

    if not high_pairs_df.empty:
        summary_lines.append("\n## Highest Pearson feature correlations")
        for _, row in high_pairs_df.head(10).iterrows():
            summary_lines.append(
                f"- {row['feature_1']} and {row['feature_2']}: {row['pearson_corr']:.4f}"
            )

    if target_corr_parts:
        summary_lines.append("\n## Largest target correlations")
        for feature, row in target_corr.head(10).iterrows():
            vals = ", ".join([f"{col}={row[col]:.4f}" for col in target_corr.columns if col != "max_abs_corr"])
            summary_lines.append(f"- {feature}: {vals}")

    (outdir / "nvda_dataset_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Done. Outputs saved to: {outdir}")
    print(f"Used dataset: {data_path}")


if __name__ == "__main__":
    main()
