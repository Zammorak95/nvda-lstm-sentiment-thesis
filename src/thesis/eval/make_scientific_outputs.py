#!/usr/bin/env python3
"""Generate thesis-ready scientific figures and tables from data/model outputs.

This command is intentionally read-only for data and model results. It creates a
report folder with publication-style PNG figures, CSV/Markdown/LaTeX tables, and
a small report index that lists every generated artifact.

Default command:
    python -m thesis.eval.make_scientific_outputs

Useful options:
    python -m thesis.eval.make_scientific_outputs --run-pytest
    python -m thesis.eval.make_scientific_outputs --dataset data/model_feed/model_dataset_clean.csv
    python -m thesis.eval.make_scientific_outputs --results-dir artifacts/models --outdir artifacts/reports/scientific_outputs
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve  # noqa: E402


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    figures: Path
    tables: Path
    logs: Path


def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in (current.parent, *current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parents[3]


def _ensure_src_on_path() -> None:
    root = _find_project_root()
    src = root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_src_on_path()

try:
    from thesis.paths import ARTIFACTS_DIR, DATA_DIR, MODELS
except Exception:
    _ROOT = _find_project_root()
    DATA_DIR = Path(os.getenv("THESIS_DATA_DIR", _ROOT / "data")).resolve()
    ARTIFACTS_DIR = Path(os.getenv("THESIS_ARTIFACTS_DIR", _ROOT / "artifacts")).resolve()
    MODELS = Path(os.getenv("THESIS_MODELS_DIR", ARTIFACTS_DIR / "models")).resolve()


DEFAULT_DATASET = DATA_DIR / "model_feed" / "model_dataset_clean.csv"
DEFAULT_RESULTS_DIR = Path(os.getenv("THESIS_MODELS_DIR", MODELS))
DEFAULT_OUTDIR = ARTIFACTS_DIR / "reports" / "scientific_outputs"
ROLLING_WINDOW = 20

FEATURE_LABELS = {
    "avg_sentiment": "Average sentiment",
    "sentiment_std": "Sentiment std.",
    "news_count": "News count",
    "target_next_return": "Next-day return",
    "target_direction": "Next-day direction",
    "log_return": "NVDA log return",
    "overnight_return": "Overnight return",
    "momentum_5d": "Momentum 5D",
    "momentum_20d": "Momentum 20D",
    "volatility_20d": "Volatility 20D",
    "volume_change": "Volume change",
    "volume_20d_avg": "Volume 20D average",
    "spy_return": "SPY return",
    "soxx_return": "SOXX return",
    "ief_return": "IEF return",
    "trends_zscore_30d": "Trends z-score 30D",
    "trends_momentum_7d": "Trends momentum 7D",
    "trends_spike": "Trends spike",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.constrained_layout.use": True,
        }
    )


def make_output_dirs(outdir: Path) -> OutputPaths:
    paths = OutputPaths(
        root=outdir,
        figures=outdir / "figures",
        tables=outdir / "tables",
        logs=outdir / "logs",
    )
    for path in (paths.root, paths.figures, paths.tables, paths.logs):
        path.mkdir(parents=True, exist_ok=True)
    return paths


def clean_name(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    return name.replace("_", " ").strip().title()


def write_table(df: pd.DataFrame, path_base: Path, float_format: str = "%.4f") -> list[Path]:
    """Write one table as CSV, Markdown, and LaTeX when available."""
    created: list[Path] = []
    csv_path = path_base.with_suffix(".csv")
    md_path = path_base.with_suffix(".md")
    tex_path = path_base.with_suffix(".tex")

    df.to_csv(csv_path, index=False)
    created.append(csv_path)

    md_path.write_text(_to_markdown(df), encoding="utf-8")
    created.append(md_path)

    try:
        tex_path.write_text(df.to_latex(index=False, float_format=float_format), encoding="utf-8")
        created.append(tex_path)
    except ImportError as exc:
        warning_path = path_base.with_suffix(".latex_warning.txt")
        warning_path.write_text(
            "LaTeX table was skipped because an optional dependency is missing.\n"
            f"Original error: {exc}\n"
            "Install with: python -m pip install jinja2\n",
            encoding="utf-8",
        )
        created.append(warning_path)
    return created


def _to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._\n"
    formatted = df.copy()
    for col in formatted.columns:
        formatted[col] = formatted[col].map(_format_cell)
    headers = [str(c).replace("|", "\\|") for c in formatted.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in formatted.iterrows():
        cells = [str(v).replace("|", "\\|") for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _format_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def find_first(root: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(root.rglob(pattern)) if root.exists() else []
        if matches:
            return matches[0]
    return None


def load_dataset(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"Dataset not found: {path}")
        return None
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
    return df


def numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = {"date", "target_direction", "target_next_return"}
    cols = [c for c in df.columns if c not in drop_cols]
    return df[cols].select_dtypes(include=[np.number]).copy()


def standardize(series: pd.Series) -> pd.Series:
    series = series.astype(float)
    std = series.std()
    if std and np.isfinite(std):
        return (series - series.mean()) / std
    return series - series.mean()


def generate_dataset_tables(df: pd.DataFrame, outputs: OutputPaths) -> list[Path]:
    created: list[Path] = []
    date_min = df["date"].min().date().isoformat() if "date" in df.columns and df["date"].notna().any() else ""
    date_max = df["date"].max().date().isoformat() if "date" in df.columns and df["date"].notna().any() else ""
    positive_rate = float(df["target_direction"].mean()) if "target_direction" in df.columns else np.nan

    overview = pd.DataFrame(
        [
            {"metric": "Rows", "value": len(df)},
            {"metric": "Columns", "value": df.shape[1]},
            {"metric": "Start date", "value": date_min},
            {"metric": "End date", "value": date_max},
            {"metric": "Positive target rate", "value": positive_rate},
            {"metric": "Missing cells", "value": int(df.isna().sum().sum())},
        ]
    )
    created += write_table(overview, outputs.tables / "dataset_overview")

    if "target_direction" in df.columns:
        counts = df["target_direction"].value_counts().reindex([0, 1], fill_value=0)
        total = int(counts.sum())
        balance = pd.DataFrame(
            [
                {
                    "class": "Down / flat",
                    "target_value": 0,
                    "observations": int(counts.loc[0]),
                    "share": float(counts.loc[0] / total) if total else np.nan,
                },
                {
                    "class": "Up",
                    "target_value": 1,
                    "observations": int(counts.loc[1]),
                    "share": float(counts.loc[1] / total) if total else np.nan,
                },
            ]
        )
        created += write_table(balance, outputs.tables / "target_class_balance")

        majority_class = int(counts.idxmax())
        majority_label = "Up" if majority_class == 1 else "Down / flat"
        baseline = pd.DataFrame(
            [
                {
                    "baseline": "Majority-class classifier",
                    "prediction_rule": f"Always predict {majority_label}",
                    "accuracy": float(counts.max() / total) if total else np.nan,
                    "auc_reference": 0.5,
                }
            ]
        )
        created += write_table(baseline, outputs.tables / "classification_baselines")

    X = numeric_features(df)
    if not X.empty:
        desc = X.describe().T.reset_index().rename(columns={"index": "feature"})
        desc["feature"] = desc["feature"].map(clean_name)
        missing = X.isna().sum().rename("missing").reset_index().rename(columns={"index": "feature"})
        missing["feature"] = missing["feature"].map(clean_name)
        desc = desc.merge(missing, on="feature", how="left")
        created += write_table(desc, outputs.tables / "feature_descriptives")

        if "target_direction" in df.columns:
            corr_target = X.corrwith(df["target_direction"]).dropna()
            target_table = (
                pd.DataFrame(
                    {
                        "feature": [clean_name(c) for c in corr_target.index],
                        "pearson_corr_with_target": corr_target.values,
                        "abs_corr_with_target": np.abs(corr_target.values),
                    }
                )
                .sort_values("abs_corr_with_target", ascending=False)
                .reset_index(drop=True)
            )
            created += write_table(target_table, outputs.tables / "feature_target_correlations")

        if X.shape[1] >= 2:
            corr = X.corr().abs()
            mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
            pairs = corr.where(mask).stack().sort_values(ascending=False).head(20)
            pair_table = pd.DataFrame(
                [
                    {
                        "feature_1": clean_name(idx[0]),
                        "feature_2": clean_name(idx[1]),
                        "abs_pearson_corr": float(value),
                    }
                    for idx, value in pairs.items()
                ]
            )
            created += write_table(pair_table, outputs.tables / "top_interfeature_correlations")

    return created


def generate_target_distribution(df: pd.DataFrame, outputs: OutputPaths) -> Path | None:
    if "target_direction" not in df.columns:
        return None
    counts = df["target_direction"].value_counts().reindex([0, 1], fill_value=0)
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    labels = ["Down / flat", "Up"]
    bars = ax.bar(labels, counts.values)
    ax.set_title("Target class distribution")
    ax.set_xlabel("Next-day direction")
    ax.set_ylabel("Observations")
    ax.grid(axis="y", alpha=0.25)

    ymax = max(counts.values) if len(counts) else 0
    ax.set_ylim(0, ymax * 1.18 if ymax else 1)
    for bar, value in zip(bars, counts.values, strict=False):
        share = value / total if total else np.nan
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(value):,}\n({share:.1%})",
            ha="center",
            va="bottom",
        )

    path = outputs.figures / "target_class_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_feature_correlation_heatmap(df: pd.DataFrame, outputs: OutputPaths) -> Path | None:
    X = numeric_features(df)
    if X.shape[1] < 2:
        return None

    if "target_direction" in df.columns:
        corr_to_target = X.corrwith(df["target_direction"]).abs().sort_values(ascending=False)
        selected_cols = corr_to_target.head(16).index.tolist()
    else:
        selected_cols = X.columns[:16].tolist()

    corr = X[selected_cols].corr()
    labels = [clean_name(c) for c in selected_cols]
    fig_size = max(7.2, 0.42 * len(selected_cols) + 2.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.82))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title("Feature correlation heatmap")
    ax.set_xticks(range(len(selected_cols)))
    ax.set_yticks(range(len(selected_cols)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    if len(selected_cols) <= 12:
        for i in range(len(selected_cols)):
            for j in range(len(selected_cols)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation")
    path = outputs.figures / "feature_correlation_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_feature_target_correlation_bar(df: pd.DataFrame, outputs: OutputPaths) -> Path | None:
    if "target_direction" not in df.columns:
        return None
    X = numeric_features(df)
    if X.empty:
        return None

    corr = X.corrwith(df["target_direction"]).dropna().sort_values(key=lambda s: s.abs(), ascending=False).head(12)
    if corr.empty:
        return None

    ordered = corr.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.barh([clean_name(c) for c in ordered.index], ordered.values)
    ax.axvline(0, linewidth=1)
    ax.set_title("Feature correlation with next-day direction")
    ax.set_xlabel("Pearson correlation with target")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.25)
    path = outputs.figures / "feature_target_correlation_bar.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_time_series_figures(df: pd.DataFrame, outputs: OutputPaths) -> list[Path]:
    if "date" not in df.columns:
        return []

    groups = {
        "returns_momentum_rolling_zscore": {
            "title": "Returns, momentum and volatility",
            "columns": ["target_next_return", "log_return", "momentum_20d", "volatility_20d"],
        },
        "sentiment_news_rolling_zscore": {
            "title": "Sentiment and news intensity",
            "columns": ["avg_sentiment", "sentiment_std", "news_count"],
        },
        "google_trends_rolling_zscore": {
            "title": "Google Trends attention variables",
            "columns": ["trends_zscore_30d", "trends_momentum_7d", "trends_spike"],
        },
    }

    created: list[Path] = []
    for filename, spec in groups.items():
        cols = [c for c in spec["columns"] if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if not cols:
            continue

        fig, ax = plt.subplots(figsize=(8.2, 4.1))
        for col in cols:
            plotted = standardize(df[col]).rolling(ROLLING_WINDOW, min_periods=max(5, ROLLING_WINDOW // 4)).mean()
            ax.plot(df["date"], plotted, linewidth=1.4, label=clean_name(col))

        ax.axhline(0, linewidth=0.8, alpha=0.5)
        ax.set_title(f"{spec['title']} ({ROLLING_WINDOW}-day rolling z-score)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Rolling standardized value")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best", frameon=False)
        path = outputs.figures / f"{filename}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        created.append(path)

    return created


def generate_random_search_outputs(results_dir: Path, outputs: OutputPaths) -> list[Path]:
    created: list[Path] = []
    path = find_first(results_dir, ["random_search_results.csv"])
    if path is None:
        return created

    rs = pd.read_csv(path)
    if rs.empty:
        return created

    created += write_table(rs.sort_values("val_auc", ascending=False).head(10), outputs.tables / "random_search_top_trials")

    if "trial" in rs.columns and "val_auc" in rs.columns:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(rs["trial"], rs["val_auc"], marker="o", linewidth=1.2, markersize=3)
        ax.axhline(rs["val_auc"].max(), linestyle="--", linewidth=1, label="Best validation AUC")
        ax.axhline(0.5, linestyle=":", linewidth=1, label="Random AUC reference")
        ax.set_title("Random search validation AUC by trial")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Validation AUC")
        ax.set_ylim(max(0.0, rs["val_auc"].min() - 0.05), min(1.0, rs["val_auc"].max() + 0.05))
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig_path = outputs.figures / "random_search_validation_auc.png"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        created.append(fig_path)

    hyper_cols = [c for c in ["lookback", "lstm_units", "dense_units", "dropout", "recurrent_dropout", "lr", "batch"] if c in rs.columns]
    if "val_auc" in rs.columns and hyper_cols:
        summary_rows = []
        for col in hyper_cols:
            grouped = rs.groupby(col)["val_auc"].agg(["count", "mean", "max"]).reset_index()
            best = grouped.sort_values(["mean", "max"], ascending=False).iloc[0]
            summary_rows.append(
                {
                    "hyperparameter": col,
                    "best_value_by_mean_auc": best[col],
                    "trials": int(best["count"]),
                    "mean_val_auc": float(best["mean"]),
                    "max_val_auc": float(best["max"]),
                }
            )
        created += write_table(pd.DataFrame(summary_rows), outputs.tables / "random_search_hyperparameter_summary")

    return created


def generate_walk_forward_outputs(results_dir: Path, outputs: OutputPaths) -> list[Path]:
    created: list[Path] = []
    oos_path = find_first(results_dir, ["walk_forward_oos_predictions.csv"])
    summary_path = find_first(results_dir, ["walk_forward_summary.json"])
    if oos_path is None:
        return created

    oos = pd.read_csv(oos_path)
    if oos.empty or not {"y_true", "y_prob_up", "y_pred"}.issubset(oos.columns):
        return created
    if "date" in oos.columns:
        oos["date"] = pd.to_datetime(oos["date"], errors="coerce")
        oos = oos.sort_values("date").reset_index(drop=True)

    metrics = []
    metrics.append({"metric": "Observations", "value": len(oos)})
    metrics.append({"metric": "Accuracy", "value": float(accuracy_score(oos["y_true"], oos["y_pred"]))})
    if len(np.unique(oos["y_true"])) > 1:
        fpr, tpr, _ = roc_curve(oos["y_true"], oos["y_prob_up"])
        metrics.append({"metric": "ROC AUC", "value": float(auc(fpr, tpr))})
    else:
        fpr, tpr = np.array([]), np.array([])
        metrics.append({"metric": "ROC AUC", "value": np.nan})
    metrics.append({"metric": "Trade rate", "value": float(oos["y_pred"].mean())})

    if "target_next_return" in oos.columns:
        strategy_returns = oos["target_next_return"].astype(float).to_numpy() * oos["y_pred"].astype(float).to_numpy()
        buy_hold_returns = oos["target_next_return"].astype(float).to_numpy()
        metrics.append({"metric": "Mean strategy return", "value": float(np.nanmean(strategy_returns))})
        metrics.append({"metric": "Mean buy-and-hold return", "value": float(np.nanmean(buy_hold_returns))})
        metrics.append({"metric": "Strategy Sharpe", "value": _sharpe(strategy_returns)})

    created += write_table(pd.DataFrame(metrics), outputs.tables / "walk_forward_oos_metrics")

    cm = confusion_matrix(oos["y_true"], oos["y_pred"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Walk-forward confusion matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks([0, 1], labels=["Down / flat", "Up"])
    ax.set_yticks([0, 1], labels=["Down / flat", "Up"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig_path = outputs.figures / "walk_forward_confusion_matrix.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    created.append(fig_path)

    if len(fpr) > 0:
        fig, ax = plt.subplots(figsize=(5.8, 4.4))
        ax.plot(fpr, tpr, linewidth=1.8, label=f"ROC AUC = {auc(fpr, tpr):.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random classifier")
        ax.set_title("Walk-forward ROC curve")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        fig_path = outputs.figures / "walk_forward_roc_curve.png"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        created.append(fig_path)

    if "date" in oos.columns:
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        ax.plot(oos["date"], oos["y_prob_up"], linewidth=1.1, label="Predicted probability of up day")
        if "threshold" in oos.columns:
            ax.plot(oos["date"], oos["threshold"], linestyle="--", linewidth=1, label="Decision threshold")
        ax.set_title("Walk-forward predicted probabilities")
        ax.set_xlabel("Date")
        ax.set_ylabel("Probability")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig_path = outputs.figures / "walk_forward_predicted_probabilities.png"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        created.append(fig_path)

    if "date" in oos.columns and "target_next_return" in oos.columns:
        strategy = oos["target_next_return"].astype(float) * oos["y_pred"].astype(float)
        buy_hold = oos["target_next_return"].astype(float)
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        ax.plot(oos["date"], (1 + strategy.fillna(0)).cumprod() - 1, linewidth=1.5, label="Model long-only strategy")
        ax.plot(oos["date"], (1 + buy_hold.fillna(0)).cumprod() - 1, linewidth=1.2, label="Buy-and-hold next-return benchmark")
        ax.set_title("Cumulative out-of-sample return")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative return")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig_path = outputs.figures / "walk_forward_cumulative_returns.png"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        created.append(fig_path)

    if summary_path is not None:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            folds = summary.get("folds", [])
            if folds:
                fold_df = pd.DataFrame(folds)
                keep = [c for c in ["fold", "train_end_date", "test_start_date", "test_end_date", "val_auc", "val_acc", "test_auc", "test_acc", "threshold"] if c in fold_df.columns]
                created += write_table(fold_df[keep], outputs.tables / "walk_forward_fold_summary")
        except Exception as exc:
            (outputs.logs / "walk_forward_summary_parse_error.txt").write_text(str(exc), encoding="utf-8")

    return created


def _sharpe(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) == 0 or returns.std() == 0:
        return float("nan")
    return float(returns.mean() / returns.std() * np.sqrt(252))


def run_pytest(outputs: OutputPaths) -> list[Path]:
    created: list[Path] = []
    log_path = outputs.logs / "pytest_output.txt"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=_find_project_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    created.append(log_path)
    summary = pd.DataFrame(
        [
            {
                "command": f"{sys.executable} -m pytest -q",
                "return_code": result.returncode,
                "passed": result.returncode == 0,
            }
        ]
    )
    created += write_table(summary, outputs.tables / "pytest_summary")
    return created


def write_report_index(created: list[Path], outputs: OutputPaths, dataset: Path, results_dir: Path) -> Path:
    rel_items = []
    for path in sorted(set(created)):
        try:
            rel_items.append(path.relative_to(outputs.root))
        except ValueError:
            rel_items.append(path)

    lines = [
        "# Scientific output index",
        "",
        f"Dataset: `{dataset}`",
        f"Results directory: `{results_dir}`",
        "",
        "## Generated files",
        "",
    ]
    if rel_items:
        lines.extend([f"- `{item}`" for item in rel_items])
    else:
        lines.append("No files were generated. Check dataset and results paths.")
    lines.append("")
    path = outputs.root / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate thesis-ready figures and tables.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--run-pytest", action="store_true", help="Run pytest and save a test summary/log.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    outputs = make_output_dirs(args.outdir)
    created: list[Path] = []

    df = load_dataset(args.dataset)
    if df is not None:
        created += generate_dataset_tables(df, outputs)
        for figure in (
            generate_target_distribution(df, outputs),
            generate_feature_correlation_heatmap(df, outputs),
            generate_feature_target_correlation_bar(df, outputs),
        ):
            if figure is not None:
                created.append(figure)
        created += generate_time_series_figures(df, outputs)

    created += generate_random_search_outputs(args.results_dir, outputs)
    created += generate_walk_forward_outputs(args.results_dir, outputs)

    if args.run_pytest:
        created += run_pytest(outputs)

    index_path = write_report_index(created, outputs, args.dataset, args.results_dir)

    print("Scientific outputs generated in:", outputs.root)
    print("Index:", index_path)
    print("Figures:", outputs.figures)
    print("Tables:", outputs.tables)
    if not created:
        print("No figures/tables were created. Check --dataset and --results-dir.")


if __name__ == "__main__":
    main()
